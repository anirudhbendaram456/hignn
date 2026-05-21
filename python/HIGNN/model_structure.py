import torch
import torch.nn as nn
from torch_scatter import scatter


class Two_body_net(torch.nn.Module):
    def __init__(self, net):
        super(Two_body_net, self).__init__()
        self.net = net

    def forward(self, x):
        # Removed with torch.no_grad() to enable gradient tracking
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        # print(r.size())
        y = self.net(x)
        y = y / r
        unflatten = nn.Unflatten(1, (3, 3))
        y = unflatten(y)

        return y


class Two_body_net_Hmatrix(torch.nn.Module):
    def __init__(self, net):
        super(Two_body_net_Hmatrix, self).__init__()
        self.net = net

    def forward(self, x):
        # Removed with torch.no_grad() to enable gradient tracking
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        y = self.net(x)
        y[:, 0] = y[:, 0] + 1.0
        y[:, 4] = y[:, 4] + 1.0
        y[:, 8] = y[:, 8] + 1.0
        y = y / r
        unflatten = nn.Unflatten(1, (3, 3))
        y = unflatten(y)
        return y


class Three_body_net(torch.nn.Module):
    def __init__(self, net):
        super(Three_body_net, self).__init__()
        self.net = net

    def forward(self, x):

        r1 = torch.norm(x[:, 0:3], p=2, dim=1, keepdim=True)
        r2 = torch.norm(x[:, 3:6], p=2, dim=1, keepdim=True)
        y = self.net(x)
        y = y / r1
        y = y / r2
        unflatten = nn.Unflatten(1, (3, 3))
        y = unflatten(y)
        return y


class Two_body_self_net(torch.nn.Module):
    def __init__(self, net):
        super(Two_body_self_net, self).__init__()
        self.net = net

    def forward(self, x):

        r = torch.norm(x, p=2, dim=1, keepdim=True)
        y = self.net(x)
        y = y / r / r
        unflatten = nn.Unflatten(1, (3, 3))
        y = unflatten(y)

        return y


# Define the forward of three
class HIGNN_mdoel(torch.nn.Module):
    def __init__(self, nn_2body, nn_3body, nn_self):
        super(HIGNN_mdoel, self).__init__()
        self.nn_2body = nn_2body
        self.nn_3body = nn_3body
        self.nn_self = nn_self

    def forward(
        self,
        x,
        edge_2body,
        edge_3body,
        edge_2bodySelf,
        edge_1body,
        edge_attr_2body,
        edge_attr_3body,
        edge_attr_2bodySelf,
        edge_attr_1body,
    ):
        # two body velocity
        xj = x[edge_2body[0, :], :]
        xi = x[edge_2body[1, :], :]

        # normalize the output of model
        mobility_2body = self.nn_2body(xj - xi)
        y_2body = torch.bmm(mobility_2body, edge_attr_2body.unsqueeze(2)).squeeze(2)
        v_2body = scatter(
            y_2body, edge_2body[1, :], dim=0, dim_size=x.size(0), reduce="add"
        )

        # three body velocity
        xj = x[edge_3body[0, :], :]
        xk = x[edge_3body[1, :], :]
        xi = x[edge_3body[2, :], :]
        x_in = torch.cat((xk - xj, xi - xk), dim=1)

        NN = x_in.size(0)
        n_max_per_GPU = 1000000
        if NN < n_max_per_GPU:
            mobility_3body = self.nn_3body(x_in)
            y_3body = torch.bmm(mobility_3body, edge_attr_3body.unsqueeze(2)).squeeze(2)
            v_3body = scatter(
                y_3body, edge_3body[2, :], dim=0, dim_size=x.size(0), reduce="add"
            )
        else:
            device = x.device
            v_3body = torch.zeros(x.size(0), 3).to(device)
            chunks = torch.arange(0, NN, n_max_per_GPU)
            edge_attr_3body = edge_attr_3body.unsqueeze(2)
            if chunks[-1] < NN:
                chunks = torch.cat((chunks, torch.tensor([NN])), dim=0)
            for i in range(chunks.size(0) - 1):
                # print(i)
                mobility_3body = self.nn_3body(x_in[chunks[i] : chunks[i + 1], :])
                y_3body = torch.bmm(
                    mobility_3body, edge_attr_3body[chunks[i] : chunks[i + 1], :, :]
                ).squeeze(2)
                v_3body = v_3body + scatter(
                    y_3body,
                    edge_3body[2, chunks[i] : chunks[i + 1]],
                    dim=0,
                    dim_size=x.size(0),
                    reduce="add",
                )

        # two body self
        xj = x[edge_2bodySelf[0, :], :]
        xi = x[edge_2bodySelf[1, :], :]
        x_in = xj - xi
        NN = x_in.size(0)
        if NN < n_max_per_GPU:
            mobility_2bodySelf = self.nn_self(x_in)
            y_2bodySelf = torch.bmm(
                mobility_2bodySelf, edge_attr_2bodySelf.unsqueeze(2)
            ).squeeze(2)
            v_2bodySelf = scatter(
                y_2bodySelf,
                edge_2bodySelf[1, :],
                dim=0,
                dim_size=x.size(0),
                reduce="add",
            )
        else:
            device = x.device
            v_2bodySelf = torch.zeros(x.size(0), 3).to(device)
            chunks = torch.arange(0, NN, n_max_per_GPU)
            edge_attr_2bodySelf = edge_attr_2bodySelf.unsqueeze(2)
            if chunks[-1] < NN:
                chunks = torch.cat((chunks, torch.tensor([NN])), dim=0)
            for i in range(chunks.size(0) - 1):
                mobility_2bodySelf = self.nn_self(x_in[chunks[i] : chunks[i + 1], :])
                y_2bodySelf = torch.bmm(
                    mobility_2bodySelf,
                    edge_attr_2bodySelf[chunks[i] : chunks[i + 1], :, :],
                ).squeeze(2)
                v_2bodySelf = v_2bodySelf + scatter(
                    y_2bodySelf,
                    edge_2bodySelf[1, chunks[i] : chunks[i + 1]],
                    dim=0,
                    dim_size=x.size(0),
                    reduce="add",
                )

        velocity = v_2body + v_3body + v_2bodySelf
        return velocity
