import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from skimage.segmentation import quickshift, mark_boundaries
from skimage import img_as_float
import matplotlib.pyplot as plt
import torchvision
from PIL import Image



# Utility function to show segmented images
def show_segmented_images(image, segments):
    plt.figure(dpi=100)
    image = img_as_float(image.permute(1, 2, 0).numpy())
    segmented_image = mark_boundaries(image, segments)

    plt.imshow(segmented_image)
    plt.title("Image with Quickshift Segments")
    plt.axis("off")
    plt.show()



def show_batch(images):
    g = torchvision.utils.make_grid(images, nrow=8, pad_value=1.0)
    g = g.numpy().transpose((1, 2, 0))
    plt.figure(dpi=200)
    plt.imshow(g)
    plt.axis("off")
    plt.show()




# Define class for conversion to PyTorch Dataset ######################################################
class ConversionToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = list(dataset)
        self.transform = transform  

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = Image.fromarray(image.numpy())
        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)
        label = torch.tensor(int(label.numpy()), dtype=torch.long)
        return image, label
    



# Utility functions ################################################################################
def create_segments(images_list, method='quickshift', kernel_size=6, max_dist=30, ratio=0.3):
    segments_list = []

    # Apply Quickshift
    for i in range(len(images_list)):
        image = images_list[i][0]
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format
        image_float = img_as_float(image) # Convert to float
        segments = quickshift(image_float, 
                              kernel_size=kernel_size, 	# kernel_size: Controls the spatial smoothing. Larger values mean more spatial influence — increasing this can merge segments, 
                                                #              but also makes boundaries less precise.
	                          max_dist=max_dist,      # max_dist: This is the maximum distance between points in the feature space (color + spatial) for them to be considered similar. 
                                                #           Increasing this reduces the number of segments by allowing more aggressive merging.
	                          ratio=ratio)        # ratio: Balances color similarity vs spatial proximity. A smaller ratio emphasizes color more, while a larger one emphasizes spatial closeness.
        segments_list.append(segments)

    if method != 'quickshift':
        pass # Implement other segmentation methods if needed
    
    return segments_list



def pred_score(image, classifier, transform = None):
    # Define DEVICE with priority: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # Image conversion and transformation
    if transform:
        image = to_pil_image(image)
        transformed_image = transform(image)

    transformed_image = transformed_image.unsqueeze(0).to(DEVICE) 
    classifier.to(DEVICE)

    classifier.eval()
    with torch.no_grad():
        output = classifier(transformed_image)
        score = output.item()
        pred = 1 if score > 0.5 else 0
    classifier.train()

    if pred == 0:
        score = 1 - score
    
    return pred, score



def pred_score_T(image, classifier, target = None, transform = None):
    # Define DEVICE with priority: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # Image conversion and transformation
    if transform:
        image = to_pil_image(image)
        transformed_image = transform(image)

    transformed_image = transformed_image.unsqueeze(0).to(DEVICE) 
    classifier.to(DEVICE)

    classifier.eval()
    with torch.no_grad():
        #time1 = time.time() 
        output = classifier(transformed_image)
        #time2 = time.time()
        #
        #print("Time taken for prediction: ", time2 - time1)

        probabilities = torch.softmax(output, dim=1)
        
        pred = torch.argmax(probabilities).item() # Predicted class

        score = probabilities[0][pred].item() # Score of predicted class

        if target is not None:
            target_score = probabilities[0][target].item() # Score of target class
        #score = output.item()
        #pred = 1 if score > 0.5 else 0
    classifier.train()

    #if pred == 0:
    #    score = 1 - score
    
    if target is not None:
        return pred, score, target_score

    return pred, score 



def remove_segment(image, segments, segment_ids, fill_technique=None):
    if fill_technique == None:
        fill = 0
    else:
        pass # Implement other fill techniques if needed 
    
    # If one segment is provided, transform into a list
    if isinstance(segment_ids, int):  
        segment_ids = [segment_ids]

    mask = np.isin(segments, segment_ids)  # Create boolean mask
    mask_tensor = torch.from_numpy(mask).to(image.device)  # Convert to tensor
    mask_tensor = mask_tensor.unsqueeze(0).expand_as(image)  # Expand to match image dimensions
    
    image_copy = image.clone()

    image_copy[mask_tensor] = fill

    return image_copy



def EdC(image, segments, segment_ids):
    # If one segment is provided, transform into a list
    if isinstance(segment_ids, int):  
        segment_ids = [segment_ids]

    mask = np.isin(segments, segment_ids)  # Create boolean mask
    mask_tensor = torch.from_numpy(mask).to(image.device)  # Convert to tensor
    mask_tensor = mask_tensor.unsqueeze(0).expand_as(image)  # Expand to match image dimensions
    
    image_copy = image.clone()

    image_copy[~mask_tensor] = 0

    return image_copy
