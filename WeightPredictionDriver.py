import torch
import flops_infra_drift.consts as consts
import torch.nn.functional as nnFunc
import torch.nn as nn
import time

from flops_infra_drift.AutoEncoderDecoder import AutoEncoderDecoder
from flops_infra_drift.Utils import (
    extractShapeInfo, autoEncode, autoDecode, get_split, unflatten_weights)
from flops_infra_drift.LSTMWeightPredictor import LSTMWeightPredictor
from flops_infra_drift.ThreadedPredictor import ThreadedPredictor

X_train = []
Y_train = []
shape_info = None

def consolidate_data(weights, server_round):
    global X_train, Y_train, shape_info

    print(
        f"WeightPredictionDriver::consolidate_data() Consolidating data for server round: {server_round}"
    )    

    # Flatten weights to 1D tensor
    flattened_weights = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )

    # TODO: Remove this section after testing
    # flattened_weights = flattened_weights[:49998]

    print(
        f"WeightPredictionDriver::consolidate_data() Flattened weights shape: {flattened_weights.shape}"
    )

    # Gather meta-data for LSTM
    if server_round == 1:
        shape_info = extractShapeInfo(weights)

    # Pad the data to make it divisible by NUM_SPLITS
    flattened_weights = nnFunc.pad(flattened_weights, pad=(0, 2), mode="constant", value=0)

    # Create dataset from weights
    X_train.append(flattened_weights)
    if server_round != 1:
        Y_train.append(flattened_weights)

    print(f"WeightPredictionDriver::consolidate_data() X_train size: {len(X_train)}")
    print(f"WeightPredictionDriver::consolidate_data() Y_train size: {len(Y_train)}")


def train_model_and_predict_weights(X_train, Y_train, weight_predictor: LSTMWeightPredictor, autoED: AutoEncoderDecoder):
    print("WeightPredictionDriver::train_model_and_predict_weights() Training model")

    X_test = torch.stack(X_train).unsqueeze(0)
    X_train = X_train[:-1]
    
    # Convert to PyTorch tensors of shape (batch, seq_len, input_dim)
    X_train = torch.stack(X_train).unsqueeze(0)
    Y_train = torch.stack(Y_train).unsqueeze(0)

    X_train = autoEncode(autoED, X_train)
    Y_train = autoEncode(autoED, Y_train)
    X_test = autoEncode(autoED, X_test)

    print(f"WeightPredictionDriver::train_model_and_predict_weights() X_train shape: {X_train.shape}")
    print(f"WeightPredictionDriver::train_model_and_predict_weights() Y_train shape: {Y_train.shape}")
    print(f"WeightPredictionDriver::train_model_and_predict_weights() X_test shape: {X_test.shape}")

    # Training setup
    optimizer = torch.optim.Adam(weight_predictor.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    # Training loop    
    for epoch in range(50):
        weight_predictor.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = weight_predictor(X_train)

        # Compute loss
        loss = loss_fn(predictions, Y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"WeightPredictionDriver::train_model_and_predict_weights() Epoch: {epoch}, Loss: {loss.item():.6f}"
            )

    print("WeightPredictionDriver::train_model_and_predict_weights() Predicting weights")
    # Predict weights
    weight_predictor.eval()
    with torch.no_grad():
        predicted_weights = weight_predictor(X_test)

    predicted_weights = autoDecode(autoED, predicted_weights)
    print(
        f"WeightPredictionDriver::train_model_and_predict_weights() Predicted weights shape: {predicted_weights.shape}"
    )
    return predicted_weights


def predict_weights_parallely(X_test, weight_predictor, autoED):
    print("WeightPredictionDriver::predict_weights_parallely() Starting parallel prediction")
    predicted_weights = torch.empty(0)

    for q in range(0, consts.NUM_SPLITS, consts.NUM_VCPUS):
        threadedPredictors = []
        for w in range(0, consts.NUM_VCPUS):
            if q+w >= consts.NUM_SPLITS:
                break
            print(f"WeightPredictionDriver::predict_weights_parallely() Preparing ThreadedPredictor for iteration: {q + w}")
            X_test_smaller = get_split(X_test, q + w)
            threadedPredictors.append(ThreadedPredictor(autoED, weight_predictor, X_test_smaller, q+w))
        
        print("WeightPredictionDriver::predict_weights_parallely() Starting threads")
        [tp.start() for tp in threadedPredictors]
        [tp.join() for tp in threadedPredictors]
        print("WeightPredictionDriver::predict_weights_parallely() Threads completed")

        for tp in threadedPredictors:
            print(f"WeightPredictionDriver::predict_weights_parallely() Appending results from thread")
            predicted_weights = torch.cat((predicted_weights, tp.result.squeeze(0)), dim=0)  

    print(f"WeightPredictionDriver::predict_weights_parallely() Parallel prediction completed, predicted weights shape: {predicted_weights.shape}")
    return predicted_weights


def d_and_c_prediction(weight_predictor, autoED, server_round):
    start = time.time()
    print("WeightPredictionDriver::d_and_c_prediction() Running divide and conquer prediction")
    global X_train, Y_train, shape_info
    
    # Pad the data to make it divisible by NUM_SPLITS
    # print(f"WeightPredictionDriver::d_and_c_prediction() Padding data to make it divisible by {consts.NUM_SPLITS}")
    # for q in range(0, len(X_train)):
    #     if X_train[q].size(0) == consts.SPLIT_SIZE:
    #         X_train[q] = nnFunc.pad(X_train[q], pad=(0, 2), mode="constant", value=0)
    #     if q < len(Y_train) and Y_train[q].size(0) == consts.SPLIT_SIZE:
    #         Y_train[q] = nnFunc.pad(Y_train[q], pad=(0, 2), mode="constant", value=0)
    for q in range(0, len(X_train)):
        print(f"X_train[{q}].size(0): {X_train[q].size(0)}")
    for q in range(0, len(Y_train)):
        print(f"Y_train[{q}].size(0): {Y_train[q].size(0)}")


    # Split the weights and perform prediction for each split
    predicted_weights = torch.empty(0)

    if (server_round == consts.CLIENT_DROP_START_ROUND):
        for q in range(0, consts.NUM_SPLITS):
            X_train_smaller = get_split(X_train, q)
            Y_train_smaller = get_split(Y_train, q)

            # predicted_weights contains the concatenated predictions
            predicted_weights = torch.cat((predicted_weights, train_model_and_predict_weights(X_train_smaller, Y_train_smaller, weight_predictor, autoED).squeeze(0)), dim=0)
            print(f"WeightPredictionDriver::d_and_c_prediction() Predicted weights shape: {predicted_weights.shape}")
    else:
        X_test = X_train
        predicted_weights = predict_weights_parallely(X_test, weight_predictor, autoED)

    # Remove padded elements
    predicted_weights = predicted_weights[:-2]
    print(f"WeightPredictionDriver::d_and_c_prediction() Predicted weights shape: {predicted_weights.shape}")

    # TODO: Enable this section after testing
    
    # Unflatten the predicted weights
    reconstructed_weights = unflatten_weights(predicted_weights, shape_info)
    reconstructed_weights = [weight.numpy() for weight in reconstructed_weights]

    # print("WeightPredictionDriver::predict_weights() Reconstructed weights' shape:")
    # displayDeepShape(reconstructed_weights)
    print(
        f"WeightPredictionDriver::predict_weights() Dummy data from reconstructed weights: {reconstructed_weights[0][1][2][3]}"
    )

    end = time.time()
    print(f"WeightPredictionDriver::predict_weights() Run time: {end - start:.4f} seconds")
    
    return reconstructed_weights
    

    #TODO: Remove this section after testing
    # return predicted_weights