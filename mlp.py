import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function):
        """
        Initialize the MLP model.

        Parameters:
        - input_size (int): Size of the input features.
        - hidden_sizes (list): List containing the sizes of hidden layers.
        - output_size (int): Size of the output layer.
        - activation_function (torch.nn.Module): Activation function for hidden layers.
        """
        super(MLP, self).__init__()

        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Create hidden layers and activations dynamically
        self.layers = nn.ModuleList()

        for i in range(len(hidden_sizes)):
            # Linear layer
            #self.layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i]))
            
            #### HE WEIGHTS INITAILIZATION
            layer = nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
            self.layers.append(layer)
            # Initialize weights using He initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            # Initialize biases to zero
            nn.init.constant_(layer.bias, 0)
            
            #### BATCH NORMALIZATION
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            
            ### DROPOUT

            self.layers.append(nn.Dropout(p=0.001))

            # Activation function (except for the last layer)
            self.layers.append(activation_function())



        # Append the ouptu layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        

    def forward(self, x):
 
        # Flatten the input
        x = x.view(-1, self.input_size)

        # Forward pass through hidden layers with activation functions
        for layer in self.layers:
            x = layer(x)

        return x.view(-1)
    

    # Training Cycle

def train_model(MLP_model, optimizer, num_epochs, train_loader, max_length):
    # Define the loss function
    criterion = nn.MSELoss()
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, max_length)  # Flatten the images
            outputs = MLP_model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass

            # Update weights using the step function of our custom ADAM optimizer
            optimizer.step()

            # Store the loss. loss.item() gets the value in a tensor. This only works for scalars.
            total_loss += loss.item()
    
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')



def evaluate_model(MLP_model, test_loader):
    # Model Evaluation
    predicted_labels = []
    true_labels = []
    predicted_rounded_labels=[]
    mse_total = 0
    total_samples = 0
    total_correct_ratio= 0
    with torch.no_grad():
        MLP_model.eval()  # Set the model to evaluation mode

        for inputs, labels in test_loader:
            # Assumes inputs are already appropriately preprocessed (e.g., flattened if necessary)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = MLP_model(inputs)

            # Store predictions and true labels
            predicted_labels.extend(outputs.cpu().numpy())  # Convert to numpy array for MSE calculation
            true_labels.extend(labels.cpu().numpy())
            rounded_outputs = round_to_nearest_quarter(outputs.cpu().numpy())
            predicted_rounded_labels.extend(rounded_outputs)

            # Calculate MSE for the current batch
            mse = np.mean((outputs.cpu().numpy() - labels.cpu().numpy()) ** 2)
            mse_total += mse * labels.size(0)  # Aggregate MSE weighted by batch size
            total_samples += labels.size(0)

        # Calculate overall MSE
        overall_mse = mse_total / total_samples
        
        # Print the overall MSE and optionally display predictions and true values
        print(f"Mean Squared Error on Test Set: {overall_mse}")
        accuracy = np.mean(np.array(predicted_rounded_labels) == np.array(true_labels))
        print(f"Accuracy on Test Set: {int(accuracy*100)}%")
        print("Predicted Labels:", predicted_labels)
        print("True Labels:", true_labels)
        print("Predicted Rounded Labels:", predicted_rounded_labels)


        ####################### Confusion Matrix ############################
        # Convert float labels to string labels
        true_labels_str = [str(label) for label in true_labels]
        predicted_rounded_labels_str = [str(label) for label in predicted_rounded_labels]
        confusion = confusion_matrix(true_labels_str, predicted_rounded_labels_str)

        # Unique sorted values of labels for axis ticks
        unique_values = sorted(set(true_labels_str).union(predicted_rounded_labels_str))
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=unique_values, yticklabels=unique_values)
        plt.title('Confusion Matrix')
        plt.ylabel('True Categories')
        plt.xlabel('Predicted Categories')
        plt.show()

def round_to_nearest_quarter(number):
    # Assuming 'number' could be a numpy array with a single value
    if isinstance(number, np.ndarray) and number.size == 1:
        number = number.item()  # Convert single-item array to scalar
    
    # Ensure the number is within the 1 to 5 range before processing
    number = np.clip(number, 1, 5)
    # Scale number to shift quarters to whole numbers, round, and rescale
    rounded_number = np.round(number * 4) / 4
    # Clip again to ensure no out-of-range values after rounding
    rounded_number = np.clip(rounded_number, 1, 5)
    return rounded_number