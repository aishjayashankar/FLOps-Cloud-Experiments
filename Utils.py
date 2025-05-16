import torch
import flops_infra_drift.consts as consts

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flops_infra_drift.AutoEncoderDecoder import AutoEncoderDecoder

def extractShapeInfo(weights):
    print("Utils::extractShapeInfo() Extracting shape info")
    shape_info = []
    for weight in weights:
        shape_info.append(weight.shape)
    return shape_info


def unflatten_weights(flat_tensor, shapes):
    print(
        f"Utils::unflatten_weights() Unflattening weights with total dim: {flat_tensor.shape[0]}"
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
        print(f"Utils::displayDeepShape() Weight {i}: {weight.shape}")


def update_results_with_predicted_weights(parameters, predicted_weights):
    print("Utils::update_results_with_predicted_weights() Updating results with predicted weights")

    weights = parameters_to_ndarrays(parameters)
    parameters_shape_info = extractShapeInfo(weights)

    flattened_parameters = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )
    print("Utils::update_results_with_predicted_weights() flattened parameters and extracted shape")

    for q in range(predicted_weights.size(0)):
        flattened_parameters[q] = predicted_weights[q]

    reconstructed_parameters = unflatten_weights(flattened_parameters, parameters_shape_info)
    reconstructed_parameters = [weight.numpy() for weight in reconstructed_parameters]

    updated_parameters = ndarrays_to_parameters(reconstructed_parameters)
    print("Utils::update_results_with_predicted_weights() Updated parameters with predicted values")
    return updated_parameters


def print_test_values(weights):
    print("Utils::print_test_values() Printing test values")
    
    flattened_weights = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )

    print(f"Utils::print_test_values() Test values:\nflattened_weights[0]: {flattened_weights[0]},\nflattened_weights[10]: {flattened_weights[10]},\nflattened_weights[100]: {flattened_weights[100]},\nflattened_weights[1000]: {flattened_weights[1000]},\nflattened_weights[10000]: {flattened_weights[10000]},\nflattened_weights[100000]: {flattened_weights[100000]},\nflattened_weights[1000000]: {flattened_weights[1000000]},\n")


def autoEncode(autoEncoderDecoder: AutoEncoderDecoder, encode_this):
    print(f"WeightPredictionDriver::autoEncode() Autoencoding weights")

    B, T, D = encode_this.shape
    encoded_val = torch.stack(
        [autoEncoderDecoder.encode(encode_this[:, t]) for t in range(T)], dim=1
    )

    print(f"WeightPredictionDriver::autoEncode() Autoencoding complete")

    return encoded_val


def autoDecode(autoEncoderDecoder: AutoEncoderDecoder, decode_this):
    print(f"WeightPredictionDriver::autoDecode() Autodecoding weights")

    # Decode the predicted weights
    reconstructed_weights = autoEncoderDecoder.decode(decode_this)

    print(f"WeightPredictionDriver::autoDecode() Autodecoding complete")

    return reconstructed_weights


def get_split(weights, iteration):
    print(f"WeightPredictionDriver::get_split() Splitting weights for iteration: {iteration}")
    weights_smaller = []

    split_start = consts.SPLIT_SIZE * iteration
    split_end = consts.SPLIT_SIZE * (iteration + 1)
    print(f"WeightPredictionDriver::get_split() Split dimensions: [{split_start}, {split_end}]")

    for q in range(0, len(weights)):        
        weights_smaller.append(weights[q][split_start:split_end])

    print(f"WeightPredictionDriver::get_split() Weights smaller shape: ({len(weights_smaller)}, {len(weights_smaller[0])})")
    return weights_smaller