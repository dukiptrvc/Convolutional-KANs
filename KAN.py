import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline
from sklearn.datasets import load_iris, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error


class ResidualActivationFunction(nn.Module):
    def __init__(self, degree=3, coefficients=10):
        super(ResidualActivationFunction, self).__init__()

        self.degree = degree
        self.coefficients = torch.nn.Parameter(torch.zeros(coefficients))
        self.w_basis = torch.nn.Parameter(torch.ones(1))
        self.w_spline = torch.nn.Parameter(torch.ones(1))
        torch.nn.init.uniform_(self.w_basis, a=-1.0, b=1.0)
        self.knots = torch.linspace(0, 1, coefficients + degree + 1).tolist()

    def forward(self, x):
        bspline = self.compute_bspline(x)
        basis_function = torch.nn.functional.silu(x)
        phi = self.w_basis * basis_function + self.w_spline * bspline

        return phi

    def compute_bspline(self, x):
        bspline_values = []
        for i in range(len(x)):
            bspline_value = BSpline(np.array(self.knots), self.coefficients.detach().numpy(), self.degree)(x[i].item())
            bspline_values.append(bspline_value)
        bspline_values = np.array(bspline_values)

        return torch.tensor(bspline_values, dtype=torch.float32, requires_grad=True)

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, spline_degree=3, spline_coefficients=10):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_degree = spline_degree
        self.spline_coefficients = spline_coefficients

        self.phi_matrix = nn.ModuleList()

        for i in range(in_dim):
            row = nn.ModuleList()

            for j in range(out_dim):
                row.append(ResidualActivationFunction(spline_degree, spline_coefficients))

            self.phi_matrix.append(row)

    def forward(self, x):
        out = torch.zeros(x.size(0), self.out_dim)

        for i in range(self.in_dim):
            input_slice = x[:, i]

            for j in range(self.out_dim):
                activation_output = self.phi_matrix[i][j](input_slice)
                out[:, j] += activation_output

        return out

class KAN(nn.Module):
    def __init__(self, layer_sizes, spline_degree=3, spline_coefficients=10):
        super(KAN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                KANLayer(
                    in_dim=layer_sizes[i],
                    out_dim=layer_sizes[i + 1],
                    spline_degree=spline_degree,
                    spline_coefficients=spline_coefficients
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_dataset(dataset_name):
    if dataset_name == "iris":
        data = load_iris()
        X = data.data
        y = data.target
        is_classification = True

    elif dataset_name == "diabetes":
        data = load_diabetes()
        X = data.data
        y = data.target
        is_classification = False

    elif dataset_name == "digits":
        data = load_digits()
        X = data.data
        y = data.target
        is_classification = True

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long if is_classification else torch.float32)

    return X, y, is_classification


def train_and_evaluate_model(X, y, is_classification, layer_sizes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KAN(layer_sizes=layer_sizes, spline_degree=3, spline_coefficients=10)

    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)

        if not is_classification:
            outputs = outputs.squeeze()

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

        if not is_classification:
            outputs = outputs.squeeze()

        if is_classification:
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predicted)
            print(f'Accuracy: {accuracy * 100:.4f}%')
        else:
            mse = mean_squared_error(y_test, outputs)
            print(f'Mean Squared Error: {mse:.4f}')

    print("Model:")
    for i in range(len(layer_sizes) - 1):
        print(f"Layer {i + 1}: Input dim {layer_sizes[i]}, Output dim {layer_sizes[i + 1]}")
    print()



layer_sizes_iris = [4, 16, 8, 4, 3]
layer_sizes_diabetes = [10, 8, 4, 2, 1]
layer_sizes_digits = [64, 8, 4, 2, 10]

for dataset, layer_sizes in zip(["iris", "diabetes", "digits"], [layer_sizes_iris, layer_sizes_diabetes, layer_sizes_digits]):
    print(f"{dataset} dataset:")
    X, y, is_classification = load_dataset(dataset)
    train_and_evaluate_model(X, y, is_classification, layer_sizes)
    print()

layer_sizes_iris = [4, 32, 16, 8, 3]
layer_sizes_diabetes = [10, 8, 4, 1]
layer_sizes_digits = [64, 16, 8, 10]

for dataset, layer_sizes in zip(["iris", "diabetes", "digits"], [layer_sizes_iris, layer_sizes_diabetes, layer_sizes_digits]):
    print(f"{dataset} dataset:")
    X, y, is_classification = load_dataset(dataset)
    train_and_evaluate_model(X, y, is_classification, layer_sizes)
    print()