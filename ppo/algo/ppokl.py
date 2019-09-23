import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from shutil import copy2


class PPOKL():
    def __init__(self,
                 actor_critic,
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

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.actor_behaviors=actor_behaviors

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kl_div_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator( advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator( advantages, self.num_mini_batch)

            for sample in data_generator:
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

                self.optimizer.zero_grad()
                #loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss = value_loss * self.value_loss_coef + action_loss 
                kl_div = 0 * torch.as_tensor(loss)
                if self.actor_behaviors is not None:
                    for behavior in self.actor_behaviors:
                        _, _, _, _, dist_b= behavior.evaluate_actions(
                                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                        high_level_action_scaling = torch.exp( - 2 * dist_b.entropy() ).detach()  # just a way to scale. Look for better ways
                        kl_div += (high_level_action_scaling * kl_divergence(dist_b, dist_a)).mean()
                    loss += kl_div * self.entropy_coef / len(self.actor_behaviors)
                else:
                    loss +=  - dist_entropy * self.entropy_coef

                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                kl_div_epoch += kl_div.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_div_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_div_epoch



def ppo_rollout(num_steps, envs, actor_critic, rollouts):
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda, use_proper_time_limits):
    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    value_loss, action_loss, dist_entropy, kl_div = agent.update(rollouts)
    rollouts.after_update()
    return value_loss, action_loss, dist_entropy, kl_div


def ppo_save_model(actor_critic, fname, iter):
    #avoid overwrite last model for safety
    torch.save(actor_critic.state_dict(), fname + ".tmp")  
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))