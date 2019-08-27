import random
import pickle
import os.path
import pandas as pd
import numpy as np

def random_pos():
    return random.randint(1, 400)/10, random.randint(1, 400)/10, random.randint(1, 400)/10

def random_size_rewards():
    return random.randint(5, 50)/10, random.randint(1, 50)/10, random.randint(1, 50)/10


def config_creator(fname, num_positive_reward, num_negative_reward, num_multi_reward, time):

    if num_positive_reward==0 and num_multi_reward==0:
        return
        

    with open(fname +".yaml", 'w+') as f:

        # Write Header
        f.write("%s \n%s \n%s \n%s \n" % ('!ArenaConfig', 'arenas:', '  0: !Arena', '    t: ' + str(time)))

        # Add items
        f.write('    items:\n')

        for ii in range(num_positive_reward):

            f.write("%s \n%s \n" % ('    - !Item', '      name: GoodGoal'))
            #specify position:
            x_pos, y_pos, z_pos = random_pos()
            y_pos = 0
            f.write("%s \n%s \n" % ('      positions:', '      - !Vector3 {x: '+str(x_pos)+', y: '+str(y_pos)+', z: '+str(z_pos)+'}'))
            #specify size
            x_siz, y_siz, z_siz = random_size_rewards()
            f.write("%s \n%s \n" % ('      sizes:', '      - !Vector3 {x: '+str(x_siz)+', y: '+str(y_siz)+', z: '+str(z_siz)+'}'))

        for nn in range(num_negative_reward):
            f.write("%s \n%s \n" % ('    - !Item', '      name: BadGoal'))
            #specify position:
            x_pos, y_pos, z_pos = random_pos()
            y_pos = 0 # change if want to have rewards on objects
            f.write("%s \n%s \n" % ('      positions:', '      - !Vector3 {x: '+str(x_pos)+', y: '+str(y_pos)+', z: '+str(z_pos)+'}'))
            #specify size
            x_siz, y_siz, z_siz = random_size_rewards()
            f.write("%s \n%s \n" % ('      sizes:', '      - !Vector3  {x: '+str(x_siz)+', y: '+str(y_siz)+', z: '+str(z_siz)+'}'))

        for nn in range(num_multi_reward):
            f.write("%s \n%s \n" % ('    - !Item', '      name: GoodGoalMulti'))
            #specify position:
            x_pos, y_pos, z_pos = random_pos()
            y_pos = 0
            f.write("%s \n%s \n" % ('      positions:', '      - !Vector3  {x: '+str(x_pos)+', y: '+str(y_pos)+', z: '+str(z_pos)+'}'))
            #specify size
            x_siz, y_siz, z_siz = random_size_rewards()
            f.write("%s \n%s \n" % ('      sizes:', '      - !Vector3  {x: '+str(x_siz)+', y: '+str(y_siz)+', z: '+str(z_siz)+'}'))

        #Create performance table

        """field_path = 'configs/generated/performance.pkl'
        if os.path.isfile(field_path):


            df = pickle.load(field_path)
            df.insert(num_positive_reward, num_negative_reward, num_multi_reward, seed, None)
            output = open(field_path,'w+')
            df.to_pickle(field_path)

        else:

            df = pd.DataFrame(columns=['Num Pos Rewards', 'Num Neg Rewards', 'Num Multi Rewards','Seed' , 'Performance'])
            next_row = df.shape[0] + 1
            df.loc[next_row] = [num_positive_reward,num_negative_reward,num_multi_reward, seed, None]
            df.to_pickle(field_path)"""




random.seed(1)
np.random.seed(1)

# for i in range(100):
#     config_creator(fname=str(i),num_positive_reward=random.randint(0, 10),
#                                 num_negative_reward=random.randint(0, 10),
#                                 num_multi_reward=random.randint(0, 10),
#                                 time=500)


