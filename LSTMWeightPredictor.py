import torch
import torch.nn as nn
import time

from flops_infra_drift.AutoED import WeightAutoencoder
from flwr.common import parameters_to_ndarrays

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


def consolidate_data(results, server_round):
    global X_train, Y_train, shape_info

    print(
        f"LSTMWeightPredictor::consolidate_data() Consolidating data for server round: {server_round}"
    )

    # Get weights of desired client
    weights = None
    for i in range(len(results)):
        if results[i][0].cid == "12569666353320040612":  # <<HARD CODED 5th Node>>
            weights = parameters_to_ndarrays(results[i][1].parameters)

    # <<TESTING>>
    weights = weights[0:12]

    # Flatten weights to 1D tensor
    flattened_weights = torch.cat(
        [torch.from_numpy(weight).flatten() for weight in weights]
    )

    # <<TESTING>>
    # flattened_weights = flattened_weights[:512]

    print(
        f"LSTMWeightPredictor::consolidate_data() Flattened weights shape: {flattened_weights.shape}"
    )

    # Gather meta-data for LSTM
    if server_round == 1:
        shape_info = extractShapeInfo(weights)

    # Create dataset from weights
    X_train.append(flattened_weights)
    # <<TESTING>>
    X_train.append(flattened_weights)
    # if server_round != 1:
    #     Y_train.append(flattened_weights)
    Y_train.append(flattened_weights)

    print(f"LSTMWeightPredictor::consolidate_data() X_train size: {len(X_train)}")
    print(f"LSTMWeightPredictor::consolidate_data() Y_train size: {len(Y_train)}")


def predict_weights():
    start = time.time()
    global X_train, Y_train, shape_info

    print("LSTMWeightPredictor::predict_weights() Predicting weights")
    # Last element of X_train is last round's weight which is used for prediction
    # Shape: (1, 1, total_dim) to account for batch size and sequence length
    X_test = torch.tensor(X_train[-1]).unsqueeze(0).unsqueeze(1)
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

    # Initialize the model
    model = LSTMWeightPredictor(
        input_dim=X_train.shape[2],
        hidden_dim=512,
        num_layers=2,
        output_dim=X_train.shape[2],
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    # Training loop
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_train)

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
    model.eval()
    with torch.no_grad():
        predicted_weights = model(X_test)

    predicted_weights = autoDecode(autoED, predicted_weights)
    print(
        f"LSTMWeightPredictor::predict_weights() Predicted weights shape: {predicted_weights.shape}"
    )

    # Unflatten the predicted weights
    reconstructed_weights = unflatten_weights(predicted_weights.squeeze(0), shape_info)
    reconstructed_weights = [weight.numpy() for weight in reconstructed_weights]

    print("LSTMWeightPredictor::predict_weights() Reconstructed weights' shape:")
    displayDeepShape(reconstructed_weights)
    print(
        f"LSTMWeightPredictor::predict_weights() Dummy data from reconstructed weights: {reconstructed_weights[0][1][2][3]}"
    )

    end = time.time()
    print(f"LSTMWeightPredictor::predict_weights() Run time: {end - start:.4f} seconds")
