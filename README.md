# Mask_Remover_APP
- implementation of transformation wearing mask face into unmasked face 
- use PHP, JAVA, Python 
- reference paper : [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)

# Demo
- https://user-images.githubusercontent.com/61732687/113242086-0c85d780-92eb-11eb-9f0e-2d1ace3688f4.mp4

# Full Flow
- ![1-1-1_Fuction_Block_ _Flow_Chart](https://user-images.githubusercontent.com/61732687/113379671-1542e000-93b5-11eb-8d89-f0cac59d4fd6.png)
- 1.Crop faces from input image.(with face-pytorch)
- 2.Mask Remover(with Mask Remover GAN)
- 3.Merge unmasked faces to original image.

# Generator
- ![2-1-3-3_Generator_구조](https://user-images.githubusercontent.com/61732687/113379789-6bb01e80-93b5-11eb-9a46-ec8cffd87e38.png)

# Discriminator
- ![2-1-3-4_Discriminator](https://user-images.githubusercontent.com/61732687/113379801-71a5ff80-93b5-11eb-8d6c-cf047c98c705.png)

# Training Flow
- ![2-1-3-1_Pre-Processing(Kaggle_Dataset_활용)](https://user-images.githubusercontent.com/61732687/113379815-81bddf00-93b5-11eb-8d36-18bbb6dc881a.png)
- ![2-1-3-2_Train_Flow(Kaggle_Dataset_활용)](https://user-images.githubusercontent.com/61732687/113379805-78347700-93b5-11eb-8595-2d0a9ca5d5a1.png)
- 1.Make the binarization image of Mask Location
- 2.Concat binarization image with original masked image
- 3.Remove the mask from face
- 4.Disriminator discriminate Generated Image to Fake and Real Image to Real

# Trainset Result
- ![3-3-7-1_train_result2](https://user-images.githubusercontent.com/61732687/113379761-576c2180-93b5-11eb-8356-72f24153a71c.png)

# Testset Result
- ![3-3-8-1_실제_마스크를_쓴_Input_Image](https://user-images.githubusercontent.com/61732687/113379775-60f58980-93b5-11eb-86e5-bd6466461e02.jpg)
- ![3-3-8-2_실제_마스크를_쓴_generated_Image](https://user-images.githubusercontent.com/61732687/113379777-6357e380-93b5-11eb-972c-1b6985884dd0.png)

