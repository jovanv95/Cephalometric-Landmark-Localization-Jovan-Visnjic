import torch
import time as time
import pandas as pd

def train(epoch_num, model, train_loader, test_loader, optimizer, criterion,scheduler):
    val_list = []  # List to store validation losses
    train_list = []  # List to store training losses
    best_val_loss = float('inf')  # Initialize the best validation loss with infinity

    # Loop over the specified number of epochs
    for epoch in range(epoch_num):
        start_time = time.time()  # Record the start time of the epoch
        model.train()  # Set the model in training mode
        model.to("cuda:0")  # Move the model to the GPU

        # Loop over the training data
        for image, heatmap_true in train_loader:
            image = image.to("cuda:0")  # Move input data to the GPU
            heatmap_true = heatmap_true.to("cuda:0")  # Move target data to the GPU

            layers = []  # List to store model layers
            for layer in model.children():
                layers.append(layer)
            first_layer = layers[1]  # Get the first layer
            layers = layers[-3:]  # Get the last three layers
            image_layer = first_layer(image)  # Pass the image through the first layer

            # Loop through the specified layers in the model
            for i, layer in enumerate(layers):
                for child in model.children():
                    for param in child.parameters():
                        param.requires_grad = False  # Set parameters of other layers to not require gradients
                for param in layer.parameters():
                    param.requires_grad = True  # Set parameters of the current layer to require gradients

                layer_params = list(layer.parameters())  # Get parameters of the current layer
                optimizer.param_groups[0]['params'] = layer_params  # Update optimizer parameters

                # Forward pass through the layer and compute the loss for this layer
                image_layer, heatmap_single_hourglass_predicted = layer(image_layer)
                loss_layer = criterion(heatmap_single_hourglass_predicted, heatmap_true)

                optimizer.zero_grad()  # Zero out the gradient buffers
                loss_layer.backward(retain_graph=True)  # Perform backward pass
                optimizer.step()  # Update the model's parameters

            # Forward pass through the entire model and compute the overall loss
            heatmap_predicted = model(image)
            loss = criterion(heatmap_predicted, heatmap_true)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        model.eval()  # Set the model in evaluation mode

        with torch.no_grad():
            val_loss = 0.0

            # Loop over the test data to calculate validation loss
            for x, y in test_loader:
                x = x.to("cuda:0")
                y = y.to("cuda:0")
                outputs = model(x)
                val_loss += criterion(outputs, y).item() * x.size(0)
            val_loss /= len(test_loader.dataset)

            scheduler.step()  # Adjust the learning rate using the scheduler

            # Print the training and validation losses for the current epoch
            print(f"Epoch {epoch+1}/{epoch_num}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            val_list.append(val_loss)  # Append validation loss to the list
            train_list.append(loss.item())  # Append training loss to the list

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"best_model.pth")
                best_val_loss = val_loss

            end_time = time.time()  # Record the end time of the epoch
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    # Create a DataFrame to store validation and training losses and save it to an Excel file
    df = pd.DataFrame({"val": val_list, "train": train_list})
    df.to_excel("training.xlsx", sheet_name='Sheet_name_1')