## Object detection module usage:

(ipython) %run animal/object_detection_module/object_collect_data.py --target_dir '/target/data/directory' to collect a dataset.

(ipython) %run animal/object_detection_module/object_train.py --data_dir '/origin/data/directory' --logs_dir '/target/logs/directory' to train a object network.


## Trained network

A trained network checkpoint can be found in ckpt_path="/workspace7/Unity3D/albert/object_module_logs/model_XXX.ckpt".

which can be loaded with the following code snippet:

```python

from animal.object_detection_module import ImpalaCNNObject

object_model= ImpalaCNNObject.load(ckpt_path)

```



#### The network expects:

 - Observations inputs with shape (batch_size, 3, 84, 84)



#### The network returns 3 things:

 - prediction of detected object in one hot encoding. Where:

        -   GoodGoal: 0
        -   BadGoal: 1
        -   GoodGoalMulti: 2
        -   Wall: 3
        -   Ramp: 4
        -   CylinderTunnel: 5
        -   WallTransparent: 6
        -   CylinderTunnelTransparent: 7
        -   Cardbox1: 8
        -   Cardbox2: 9
        -   UObject: 10
        -   LObject: 11
        -   LObject2: 12
        -   DeathZone: 13
        -   HotZone: 14