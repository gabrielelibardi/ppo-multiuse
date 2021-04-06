import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from shutil import copy2
from copy import deepcopy


class PPOKL():
    def __init__(self,
                 actor_critic,
                 rew_fun,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 actor_behaviors=None):

        self.actor_critic = actor_critic
        self.rew_fun = rew_fun

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.actor_behaviors = actor_behaviors

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.meta_optimizer = optim.Adam(self.rew_fun.parameters(), lr=lr, eps=eps)
        
    def change_reward(self, rollouts):
        ex_rollouts = deepcopy(rollouts)
        for work_num in range(ex_rollouts.obs.shape[1]):
            ex_rollouts.rewards[:,work_num,:] = self.rew_fun(ex_rollouts.obs[:-1,work_num,:], ex_rollouts.obs[:-1,work_num,:], ex_rollouts.actions[:,work_num,:])
        return ex_rollouts
    
    def sample_update(self, sample):
        
        # this is the computation of the ppo loss, which is common to both the in_sample and ex_sample
        
        obs_batch, recurrent_hidden_states_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, _, dist_a = self.actor_critic.evaluate_actions(
            obs_batch, recurrent_hidden_states_batch, masks_batch,actions_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()
            
        #loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
        loss = value_loss * self.value_loss_coef + action_loss 
        kl_div = 0 * torch.as_tensor(loss)
        if self.actor_behaviors is not None:
            for behavior in self.actor_behaviors:
                with torch.no_grad():
                    _, _, _, _, dist_b= behavior.evaluate_actions(
                                obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                    high_level_action_scaling = torch.exp( - 2 * dist_b.entropy() ).detach()  # just a way to scale. Look for better ways
                kl_div += (high_level_action_scaling * kl_divergence(dist_b, dist_a)).mean()
            loss += kl_div * self.entropy_coef / len(self.actor_behaviors)
        else:
            loss +=  - dist_entropy * self.entropy_coef

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                    self.max_grad_norm)
        return  value_loss, action_loss, dist_entropy, kl_div

    def update(self, in_rollouts, ex_rollouts):
        # Advantages are computed for both rollouts and are kept separate
        in_advantages = in_rollouts.returns[:-1] - in_rollouts.value_preds[:-1]
        in_advantages = (in_advantages - in_advantages.mean()) / (
            in_advantages.std() + 1e-5)
        
        ex_advantages = ex_rollouts.returns[:-1] - ex_rollouts.value_preds[:-1]
        ex_advantages = (ex_advantages - ex_advantages.mean()) / (
            ex_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kl_div_epoch = 0

        for e in range(self.ppo_epoch):
            # From the 2 advantages 2 generators are created from which we sample simultaneusly ex_sample and in_sample
            if self.actor_critic.is_recurrent:
                data_generator_in = in_rollouts.recurrent_generator(in_advantages, self.num_mini_batch)
            else:
                data_generator_in = in_rollouts.feed_forward_generator(in_advantages, self.num_mini_batch)
                
            if self.actor_critic.is_recurrent:
                data_generator_ex = ex_rollouts.recurrent_generator(ex_advantages, self.num_mini_batch)
            else:
                data_generator_ex = ex_rollouts.feed_forward_generator(ex_advantages, self.num_mini_batch)

            for ex_sample, in_sample in zip(data_generator_ex, data_generator_in):
                
                self.optimizer.zero_grad()
                self.meta_optimizer.zero_grad()
                
                # compute loss first for intrinsic reward then external reward update of eta with meta_optimizer 
                value_loss, action_loss, dist_entropy, kl_div = self.sample_update(in_sample)
                self.optimizer.step()
                
                value_loss, action_loss, dist_entropy, kl_div = self.sample_update(ex_sample)
                self.meta_optimizer.step()
                #para_tensor = list(self.rew_fun.parameters())
                #print(para_tensor[0].grad)
                #print(para_tensor[1].grad)

                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                kl_div_epoch += kl_div.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_div_epoch /= num_updates
        mean_loss = value_loss_epoch * self.value_loss_coef +  action_loss_epoch 
        mean_loss += dist_entropy_epoch*self.entropy_coef if self.actor_behaviors is None else kl_div_epoch*self.entropy_coef 
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_div_epoch, mean_loss


def ppo_rollout(num_steps, envs, actor_critic, rollouts, det=False):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step], deterministic = det)

        # Obser reward and next obs
#         action = action * 0
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda, use_proper_time_limits, rew_func):
    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    # create 2 rollouts one for the intrinsic rewards, the other for the extrinsic rewards
    # later we can make it more efficient, for state-input envs it makes almost no difference
    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    # create intrinisc reward rollout and compute returns
    in_rollouts = agent.change_reward(rollouts)
    in_rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    # the update uses both rollouts
    value_loss, action_loss, dist_entropy, kl_div, loss = agent.update(in_rollouts, rollouts)
    rollouts.after_update()
    in_rollouts.after_update()
    return value_loss, action_loss, dist_entropy, kl_div, loss


def ppo_save_model(actor_critic, fname, iter):
    #avoid overwrite last model for safety
    torch.save(actor_critic.state_dict(), fname + ".tmp")  
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))


