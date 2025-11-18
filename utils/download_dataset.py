import os

def download_tiny_imagenet_colab(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'tiny-imagenet-200')
    if os.path.exists(dataset_path):
        print(f'Dataset already exists in {dataset_path}')
        return
    
    print('Downloading Tiny-ImageNet...')
    os.system('!wget -q --show-progress http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    
    print('\nExtracting dataset...')
    os.system('!unzip -q tiny-imagenet-200.zip -d {data_dir}')
    
    print('\nRemoving zip...')
    os.system('!rm tiny-imagenet-200.zip')
    
    print(f'Dataset ready at: {dataset_path}')

if __name__ == '__main__':
    download_tiny_imagenet_colab()