import importlib.util
import glob
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig


def main():
    # Load the agent from the submission
    print('Loading your agent')
    try:
        spec = importlib.util.spec_from_file_location('agent_module', '/aaio/agent.py')
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        submitted_agent = agent_module.Agent(device='cuda')
    except Exception as e:
        print('Your agent could not be loaded, make sure all the paths are absolute, error thrown:')
        raise e
    print('Agent successfully loaded')

    arenas = glob.glob('/aaio/test/train_arenas/*.yaml')
    
    #arena_config_in = ArenaConfig('/aaio/test/1-Food.yaml')

    #print('Resetting your agent')
    #try:
    #    submitted_agent.reset(t=arena_config_in.arenas[0].t)
    #except Exception as e:
    #    print('Your agent could not be reset:')
    #    raise e

    try:
        resolution = submitted_agent.resolution
        assert resolution == 84
    except AttributeError:
        resolution = 84
    except AssertionError:
        print('Resolution must be 84 for testing')
        return

    env = AnimalAIEnv(
        environment_filename='/aaio/test/env/AnimalAI',
        seed=0,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=True,
        resolution=resolution
    )

    print('Running arenas')
    total_reward = 0
    for count,a in enumerate(arenas):
        arena_config_in = ArenaConfig(a)
        env.reset(arenas_configurations=arena_config_in)
        cumulated_reward = 0

        try:
            submitted_agent.reset(t=arena_config_in.arenas[0].t)
        except Exception as e:
            print('Agent reset failed during episode {}'.format(k))
            raise e
        try:
            obs, reward, done, info = env.step([0, 0])
            for i in range(arena_config_in.arenas[0].t):

                action = submitted_agent.step(obs, reward, done, info)
                obs, reward, done, info = env.step(action)
                cumulated_reward += reward
                if done:
                    break
        except Exception as e:
            print('Episode {} failed'.format(a))
            raise e
        total_reward += cumulated_reward
        print('Episode {0} completed, reward {1}'.format(a, cumulated_reward))

    print('SUCCESS {}'.format(total_reward/count))


if __name__ == '__main__':
    main()
