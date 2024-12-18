from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

# def _gather_feat(feat, ind, mask=None):
#     dim  = feat.size(2)
#     ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#     feat = feat.gather(1, ind)

# original
def _gather_feat(feat, ind, mask=None):
    batch_size, num_indices, dim = ind.size(0), ind.size(1), feat.size(2)

    # Преобразуем индексы для плоского индексирования
    ind = ind + torch.arange(batch_size, device=feat.device).unsqueeze(1) * feat.size(1)
    ind = ind.view(-1)  # Разворачиваем индексы для работы с плоскими тензорами

    # Разворачиваем `feat` для плоского индексирования
    feat_flat = feat.view(-1, dim)

    # Выбираем элементы по индексам
    feat = feat_flat[ind].view(batch_size, num_indices, dim)
    return feat
  
# # orig
# def _gather_feat(feat, ind, mask=None):
#     dim  = feat.size(2)

#     ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#     # ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

#     if (ind.size(2) == 2):
#         # feat = feat.gather(1, ind)
#         feat = torch.cat([torch.take(feat[:,:,:1], ind[:,:,:1]), torch.take(feat[:,:,1:], ind[:,:,1:])], dim=2)
#     else:
#         feat = torch.take(feat, ind)

#     # if mask is not None:
#     #     mask = mask.unsqueeze(2).expand_as(feat)
#     #     feat = feat[mask]
#     #     feat = feat.view(-1, dim)
#     return feat

# orig
def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

# def _gather_feat(feat, ind, mask=None):
#     dim  = feat.size(2)
#     ind_size_0 = ind.size(0)
#     ind_size_1 = ind.size(1)
    
#     if (feat.size(2) == 1):
#         dim = 1

#     ind  = ind.unsqueeze(2).expand(ind_size_0, ind_size_1, dim)
#     # ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

#     if (ind.size(2) == 2):
#         # feat = feat.gather(1, ind)
#         feat = torch.cat([torch.take(feat[:,:,:1], ind[:,:,:1]), torch.take(feat[:,:,1:], ind[:,:,1:])], dim=2)
#     else:
#         feat = torch.take(feat, ind)

#     # if mask is not None:
#     #     mask = mask.unsqueeze(2).expand_as(feat)
#     #     feat = feat[mask]
#     #     feat = feat.view(-1, dim)
#     return feat

# def _transpose_and_gather_feat(feat, ind):
#     feat = feat.permute(0, 2, 3, 1).contiguous()
#     feat_size_0 = feat.size(0)
#     feat_size_3 = feat.size(3)
#     feat = feat.view(feat_size_0, -1, feat_size_3) #-1
#     feat = _gather_feat(feat, ind)
#     return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)