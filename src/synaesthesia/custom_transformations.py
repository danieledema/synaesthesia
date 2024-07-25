from torchvision.transforms import v2

# Define a list of transformations you want to apply to your data
transformations_list = [
    v2.RandomRotation(degrees=20),
    v2.GaussianNoise(mean=0, sigma=0.1, clip=False),
    ]            
# Pass the list of transformations to Compose function
transformations = v2.RandomApply(transformations_list, p=0.7)


np.random.normal(loc=mean, scale=std_dev, size=num_elements)