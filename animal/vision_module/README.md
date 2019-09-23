## Vision module usage:

RUN python vision_collect_data.py --target_dir ´/target/data/directory´ to collect a dataset.

RUN python vision_train.py --data_dir ´/origin/data/directory´ --logs_dir ´/target/logs/directory´ to train a vision network.


## Trained network

A trained network checkpoint can be found in ckpt_path="/workspace7/Unity3D/albert/vision_module_logs/model_249.ckpt".

which can be loaded with the following code snippet:

```python

from animal.vision_module import ImpalaCNNVision

vision_model= ImpalaCNNVision.load(ckpt_path)


```



#### The network expects:

 - Observations inputs with shape (batch_size, 3, 84, 84)



#### The network returns 3 things:

 - Position prediction of shape (batch_size, 3), with the first 2 positions being x and y coordinates respectively (in range [0, 40]) and the last position being the rotation angle in degrees [0, 360].

 - RNN inner state if the network has been trained with recurrence (No the case for the network in ckpt_path).

 - deep features before the position prediction (output of the cnn).