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

