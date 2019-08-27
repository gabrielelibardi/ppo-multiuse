import importlib.util
import glob
import argparse
from shutil import copyfile

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig


def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
    '--arenas-pattern', default='*.yaml', help='pattern for glob')
    parser.add_argument(
    '--use-mine',action='store_true',default=False,help='If to use the agent.py and state_dict from host (otherwise the containered are used')
    args = parser.parse_args()
    if args.use_mine:
        copyfile('/myaaio/agent.py','/aaio/agent.py')
        copyfile('/myaaio/data/animal.state_dict','/aaio/data/animal.state_dict')
        #note these changes are lost when restarting the container

    # Load the agent from the submission
    print('Loading your agent')
    try:
        spec = importlib.util.spec_from_file_location('agent_module', '/aaio/agent.py')
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        submitted_agent = agent_module.Agent()
    except Exception as e:
        print('Your agent could not be loaded, make sure all the paths are absolute, error thrown:')
        raise e
    print('Agent successfully loaded')

    arenas = glob.glob(arenas_pattern)

    env = AnimalAIEnv(
        environment_filename='/aaio/test/env/AnimalAI',
        seed=0,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=True,
    )

    for a in arenas:
        arena_config_in = ArenaConfig(a)
        env.reset(arenas_configurations=arena_config_in)
        print('Resetting your agent')
        try:
            submitted_agent.reset(t=arena_config_in.arenas[0].t)
        except Exception as e:
            print('Your agent could not be reset:')
            raise e

        print('Running 5 episodes')

        for k in range(5):
            cumulated_reward = 0
            print('Episode {} starting'.format(k))
            try:
                obs, reward, done, info = env.step([0, 0])
                for i in range(arena_config_in.arenas[0].t):
                    
                    action = submitted_agent.step(obs, reward, done, info)
                    obs, reward, done, info = env.step(action)
                    cumulated_reward += reward
                    if done:
                        break
            except Exception as e:
                print('Episode {} failed'.format(k))
                raise e
            print('Episode {0} completed, reward {1}'.format(k, cumulated_reward))

        print('Arena {0} completed, avg reward {1}'.format(a, cumulated_reward/5))

    print('SUCCESS')


if __name__ == '__main__':
    main()
