# Non-targeted-Attack-IJCAI2019-ColdRiver
No.5 solution to non-targeted attack in IJCAI-2019 Alibaba Adversarial AI Challenge (AAAC 2019))  
IJCAI-2019 Alibaba Adversarial AI Challenge (AAAC 2019)): https://tianchi.aliyun.com/competition/entrance/231701/introduction  
We attend IJCAI-2019 Alibaba Adversarial AI Challenge, and get the 5th place in the non-targted attack track.
Our method is gradiend-based attack method.  
I use lots of tricks to improve the attack ability and transferability.
  
Currently only the scripts are released.

#### Scripts of Repo  
1. attack_tijiao2.py:  Main script for attack.
2. test_search.py: script for test the attack method
3. gen_attack.py: script to generate adversarial data for following training
4. train_ads.py: script to train adversarial model.

#### requirement
```
Python 3
pytorch 0.4 +
other necessary package used in script
```

#### Usage
To attack a model and generate adversarial images.
```
python attack_tijiao2.py --input_dir=/path/to/your/input_images --output_dir=/path/to/your/output_dir 
```
You need to replace the pretrained weights in the attack_tijiao2.py, and place the dev.csv in the input_images. 

To test the adversarial image
```
python test_search.py --input_dir=/path/to/your/input_images --output_dir=/path/to/your/output_dir --if_attack=0
```

To search for parameters of attacking, modify the script test_search.py

### Our method
You can find all of the thick in the attack_tijiao2.py  
Our method is gradient-based attack method.  
Thanks to previous work, our method based on [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884). And we add lots of our thicks, and I believe they do work.  
1. Iterative gradient ascend.  (Loss function is CrossEntropyLoss)    
2. Add Gaussian kernel convolution (Key point in the paper <Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks>)     
3. Add input diversity (resize and padding for picture) (It seems it doesn't work sometimes)  
4. Add Class Activation Map Mask for noise.  
5. Add Reverse Cross Entropy Loss to the original Loss function.  
6. Multiply pixel norm of noise to noise  
7. Ensemble model, and apply different weight for different models according to the model prediction during attack iterations.  
8. Just make the noise in the edge equal to zero  (may work)  

I'm not sure these thicks always work, I also test in imagenet(NIPS 2017 adversarial competition test dataset). But the result still not clear.

