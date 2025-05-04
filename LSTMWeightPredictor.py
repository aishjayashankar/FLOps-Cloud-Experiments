import torch
import torch.nn as nn
import time
import torch.nn.functional as nnFunc
import flops_infra_drift.consts as consts

from flops_infra_drift.AutoED import WeightAutoencoder
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

X_train = []
Y_train = []
shape_info = None


class LSTMWeightPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, output_dim=None):
        super(LSTMWeightPredictor, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output Layer (Linear)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if output_dim
            else nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)  # LSTM output, hidden state, and cell state

        # We take the output from the last time step (last token in sequence)
        out = lstm_out[:, -1, :]  # Shape: (batch, hidden_dim)
        return self.output_proj(out)  # Predict next weights (output_dim)


def autoEncode(autoED, encode_this):
    print(f"LSTMWeightPredictor::autoEncode() Autoencoding weights")

    B, T, D = encode_this.shape
    encoded_val = torch.stack(
        [autoED.encode(encode_this[:, t]) for t in range(T)], dim=1
    )

    print(f"LSTMWeightPredictor::autoEncode() Autoencoding complete")

    return encoded_val


def autoDecode(autoED, predicted_weights):
    print(f"LSTMWeightPredictor::autoDecode() Autodecoding weights")

    # Decode the predicted weights
    reconstructed_weights = autoED.decode(predicted_weights)

    print(f"LSTMWeightPredictor::autoDecode() Autodecoding complete")

    return reconstructed_weights


def extractShapeInfo(weights):
    print("LSTMWeightPredictor::extractShapeInfo() Extracting shape info")
    shape_info = []
    for weight in weights:
        shape_info.append(weight.shape)
    return shape_info


def unflatten_weights(flat_tensor, shapes):
    print(
        f"LSTMWeightPredictor::unflatten_weights() Unflattening weights with total dim: {flat_tensor.shape[0]}"
    )
    reconstructed = []
    idx = 0
    for shape in shapes:
        numel = torch.tensor(shape).prod().item() if shape != () else 1
        chunk = flat_tensor[idx : idx + numel]
        reconstructed.append(chunk.reshape(shape))
        idx += numel
    return reconstructed


def displayDeepShape(weights):
    for i, weight in enumerate(weights):
        print(f"LSTMWeightPredictor::displayDeepShape() Weight {i}: {weight.shape}")


def consolidate_data(weights, server_round):
    global X_train, Y_train, shape_info

    print(
        f"LSTMWeightPredictor::consolidate_data() Consolidating data for server round: {server_round}"
    )    

    # Flatten weights to 1D tensor
    flattened_weights = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )

    # TODO: Remove this section after testing
    # flattened_weights = flattened_weights[:49998]

    print(
        f"LSTMWeightPredictor::consolidate_data() Flattened weights shape: {flattened_weights.shape}"
    )

    # Gather meta-data for LSTM
    if server_round == 1:
        shape_info = extractShapeInfo(weights)

    # Create dataset from weights
    X_train.append(flattened_weights)
    if server_round != 1:
        Y_train.append(flattened_weights)

    print(f"LSTMWeightPredictor::consolidate_data() X_train size: {len(X_train)}")
    print(f"LSTMWeightPredictor::consolidate_data() Y_train size: {len(Y_train)}")


def get_split(weights, iteration):
    print(f"LSTMWeightPredictor::get_split() Splitting weights for iteration: {iteration}")
    weights_smaller = []

    split_start = consts.SPLIT_SIZE * iteration
    split_end = consts.SPLIT_SIZE * (iteration + 1)
    print(f"LSTMWeightPredictor::get_split() Split dimensions: [{split_start}, {split_end}]")

    for q in range(0, len(weights)):        
        weights_smaller.append(weights[q][split_start:split_end])

    print(f"LSTMWeightPredictor::get_split() Weights smaller shape: ({len(weights_smaller)}, {len(weights_smaller[0])})")
    return weights_smaller


def predict_weights(X_train, Y_train, weight_predictor: LSTMWeightPredictor, server_round):
    print("LSTMWeightPredictor::predict_weights() Predicting weights")

    X_test = torch.stack(X_train).unsqueeze(0)
    X_train = X_train[:-1]
    
    # Convert to PyTorch tensors of shape (batch, seq_len, input_dim)
    X_train = torch.stack(X_train).unsqueeze(0)
    Y_train = torch.stack(Y_train).unsqueeze(0)

    autoED = WeightAutoencoder(input_dim=X_train.shape[2], latent_dim=512)

    X_train = autoEncode(autoED, X_train)
    Y_train = autoEncode(autoED, Y_train)
    X_test = autoEncode(autoED, X_test)

    print(f"LSTMWeightPredictor::predict_weights() X_train shape: {X_train.shape}")
    print(f"LSTMWeightPredictor::predict_weights() Y_train shape: {Y_train.shape}")
    print(f"LSTMWeightPredictor::predict_weights() X_test shape: {X_test.shape}")

    # # Initialize the model
    # model = LSTMWeightPredictor(
    #     input_dim=X_train.shape[2],
    #     hidden_dim=512,
    #     num_layers=2,
    #     output_dim=X_train.shape[2],
    # )
    # If model has already been trained just predict the next set of weights
    if server_round > consts.CLIENT_DROP_START_ROUND:
        print(
            f"LSTMWeightPredictor::predict_weights() Server round: {server_round}, using trained model for prediction"
        )
        weight_predictor.eval()
        with torch.no_grad():
            predicted_weights = weight_predictor(X_test)
        predicted_weights = autoDecode(autoED, predicted_weights)
        print(
            f"LSTMWeightPredictor::predict_weights() Predicted weights shape: {predicted_weights.shape}"
        )
        return predicted_weights

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
                f"LSTMWeightPredictor::predict_weights() Epoch: {epoch}, Loss: {loss.item():.6f}"
            )

    
    # Predict weights
    weight_predictor.eval()
    with torch.no_grad():
        predicted_weights = weight_predictor(X_test)

    predicted_weights = autoDecode(autoED, predicted_weights)
    print(
        f"LSTMWeightPredictor::predict_weights() Predicted weights shape: {predicted_weights.shape}"
    )
    return predicted_weights


def d_and_c_prediction(weight_predictor, server_round):
    start = time.time()
    print("LSTMWeightPredictor::d_and_c_prediction() Running divide and conquer prediction")
    global X_train, Y_train, shape_info
    
    # Pad the data to make it divisible by NUM_SPLITS
    print(f"LSTMWeightPredictor::d_and_c_prediction() Padding data to make it divisible by {consts.NUM_SPLITS}")
    for q in range(0, len(X_train)):
        X_train[q] = nnFunc.pad(X_train[q], pad=(0, 2), mode="constant", value=0)
        if q < len(Y_train):
            Y_train[q] = nnFunc.pad(Y_train[q], pad=(0, 2), mode="constant", value=0)

    # Split the weights and perform prediction for each split
    predicted_weights = torch.empty(0)
    for q in range(0, consts.NUM_SPLITS):
        X_train_smaller = get_split(X_train, q)
        Y_train_smaller = get_split(Y_train, q)

        # predicted_weights contains the concatenated predictions
        predicted_weights = torch.cat((predicted_weights, predict_weights(X_train_smaller, Y_train_smaller, weight_predictor, server_round).squeeze(0)), dim=0)
        print(f"LSTMWeightPredictor::d_and_c_prediction() Predicted weights shape: {predicted_weights.shape}")

    # Remove padded elements
    predicted_weights = predicted_weights[:-2]

    # TODO: Enable this section after testing
    # Unflatten the predicted weights
    reconstructed_weights = unflatten_weights(predicted_weights, shape_info)
    reconstructed_weights = [weight.numpy() for weight in reconstructed_weights]

    # print("LSTMWeightPredictor::predict_weights() Reconstructed weights' shape:")
    # displayDeepShape(reconstructed_weights)
    print(
        f"LSTMWeightPredictor::predict_weights() Dummy data from reconstructed weights: {reconstructed_weights[0][1][2][3]}"
    )

    end = time.time()
    print(f"LSTMWeightPredictor::predict_weights() Run time: {end - start:.4f} seconds")
    
    return reconstructed_weights
    

    #TODO: Remove this section after testing
    # return predicted_weights

def update_results_with_predicted_weights(parameters, predicted_weights):
    print("LSTMWeightPredictor::update_results_with_predicted_weights() Updating results with predicted weights")

    weights = parameters_to_ndarrays(parameters)
    parameters_shape_info = extractShapeInfo(weights)

    flattened_parameters = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )
    print("LSTMWeightPredictor::update_results_with_predicted_weights() flattened parameters and extracted shape")

    for q in range(predicted_weights.size(0)):
        flattened_parameters[q] = predicted_weights[q]

    reconstructed_parameters = unflatten_weights(flattened_parameters, parameters_shape_info)
    reconstructed_parameters = [weight.numpy() for weight in reconstructed_parameters]

    updated_parameters = ndarrays_to_parameters(reconstructed_parameters)
    print("LSTMWeightPredictor::update_results_with_predicted_weights() Updated parameters with predicted values")
    return updated_parameters

def print_test_values(weights):
    print("LSTMWeightPredictor::print_test_values() Printing test values")
    
    flattened_weights = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )

    print(f"LSTMWeightPredictor::print_test_values() Test values:\nflattened_weights[0]: {flattened_weights[0]},\nflattened_weights[10]: {flattened_weights[10]},\nflattened_weights[100]: {flattened_weights[100]},\nflattened_weights[1000]: {flattened_weights[1000]},\nflattened_weights[10000]: {flattened_weights[10000]},\nflattened_weights[100000]: {flattened_weights[100000]},\nflattened_weights[1000000]: {flattened_weights[1000000]},\n")

