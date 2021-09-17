# UFA-Under-Supported-Feature-Augmentation-with-Pytorch

### 1. Dataset Download

The downloaded dataset should be placed in the path specified below.

 #### - MiniImageNet

- Download dataset here : 

- Place the dataset in this path

  ```
  ./miniImageNet/
  ```

#### - TieredImageNet

- Download dataset here : https://drive.google.com/file/d/1CEkp1B-bN9VR3VJDZ0qcrBRxL5KjxTTw/view?usp=sharing

- Place the dataset in this path

  ```
  ./tieredImageNet/
  ```

</br>

### 2. Pre-trainin Model

#### - MiniImageNet

```
python ./src/train.py -c ./configs/pretrain_mini.config
```

#### - TieredImageNet

```
python ./src/train.py -c ./configs/pretrain_tiered.config
```

</br>

### 3. Evaluate

#### -MiniImageNet

- ##### {Baseline}

  ```
  python ./src/train.py -c ./configs/mini_baseline.config
  ```

- ##### {Baseline}$_{rrc}$

  ```
  python ./src/train.py -c ./configs/mini_baseline_rrc.config
  ```

- ##### {Baseline}$_{rotation}$

  ```
  python ./src/train.py -c ./configs/mini_baseline_rotation.config
  ```

- ##### {Baseline}$_{rrc+rotation}$

  ```
  python ./src/train.py -c ./configs/mini_baseline_rrc+rotation.config
  ```

- ##### {UFA}$_{A_{U}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA_AU.config
  ```

- ##### {UFA (+clone)}$_{A_{U}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA+clone_AU.config
  ```

- ##### {UFA}$_{A_{R}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA_AR.config
  ```

- ##### {UFA (+clone)}$_{A_{R}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA+clone_AR.config
  ```

- ##### {UFA}$_{A_{L}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA_AL.config
  ```

- ##### {UFA (+clone)}$_{A_{L}}$

  ```
  python ./src/train.py -c ./configs/mini_UFA+clone_AL.config
  ```

</br>

#### - TieredImageNet

- ##### {Baseline}

  ```
  python ./src/train.py -c ./configs/tiered_baseline.config
  ```

- ##### {Baseline}$_{rrc}$

  ```
  python ./src/train.py -c ./configs/tiered_baseline_rrc.config
  ```

- ##### {Baseline}$_{rotation}$

  ```
  python ./src/train.py -c ./configs/tiered_baseline_rotation.config
  ```

- ##### {Baseline}$_{rrc+rotation}$

  ```
  python ./src/train.py -c ./configs/tiered_baseline_rrc+rotation.config
  ```

- ##### {UFA}$_{A_{U}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA_AU.config
  ```

- ##### {UFA (+clone)}$_{A_{U}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA+clone_AU.config
  ```

- ##### {UFA}$_{A_{R}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA_AR.config
  ```

- ##### {UFA (+clone)}$_{A_{R}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA+clone_AR.config
  ```

- ##### {UFA}$_{A_{L}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA_AL.config
  ```

- ##### {UFA (+clone)}$_{A_{L}}$

  ```
  python ./src/train.py -c ./configs/tiered_UFA+clone_AL.config
  ```

  
