import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

import DataVis
import numpy as np
import os

from tensorboardX import SummaryWriter

from tools import Ops
from tools.PointCloudDataset import save_torch_pc
from tools.PointCloudDataset import write_image_pc

class AutoEncoderTrainer(object):

    def __init__(self, model, loader, optimizer, loss_fn):

        self.optimizer = optimizer
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.logger = DataVis.LossLogger(model.name)

        self.model.cuda()


    def train(self, n_epochs, save_step=5):

        for epoch in range(n_epochs):
            out_data = None

            for i, data in enumerate(self.loader, 0):
                in_data = Variable(data.cuda())


                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                loss = self.loss_fn(out_data, in_data)
                
                self.logger.update(loss)

                loss.backward()
                self.optimizer.step()

            if epoch % save_step == 0:
                self.model.save("checkpoint", epoch)

                results_dir = os.path.join("results", self.model.name, 
                    "epoch_{}".format(str(epoch).zfill(4)))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                try:
                    self.model.save_results(results_dir, out_data)
                except NotImplementedError:
                    print "Intermediate results not saved."


class ImageToPCTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn,
            log_dir="log"):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.log_dir = log_dir
        #self.logger = DataVis.LossLogger(model.name)
        #self.val_logger = DataVis.LossLogger("Validation-"+model.name)
        #self.train_logger = DataVis.LossLogger("Training acc. "+model.name)

        self.writer = SummaryWriter(self.log_dir)

        self.model.cuda()


    def train(self, n_epochs, save_step=1):

        best_acc = 0
        it_num = 0

        for epoch in range(n_epochs):
            out_data = None
            in_data = None

            for i, data in enumerate(self.train_loader, 0):
                if data[1].size()[0] != self.model.batch_size:
                    continue

                in_data = Variable(data[0].cuda())
                target = Variable(data[1]).cuda()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                #loss = F.mse_loss(out_data, target)
                loss = self.loss_fn(out_data, target)
                print loss

                self.writer.add_scalar('train_loss', loss, it_num)
                #self.logger.update(loss)

                loss.backward()
                #from IPython import embed; embed(); exit(-1)
                self.optimizer.step()
                it_num += 1

            if epoch % save_step == 0:
                self.model.save("checkpoint", epoch)

                results_dir = os.path.join("results", self.model.name, 
                    "epoch_{}".format(str(epoch).zfill(4)))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                try:
                    self.model.save_results(results_dir, out_data)
                except NotImplementedError:
                    print "Results not saved."


            #epoch_acc = self.update_validation_accuracy(epoch)
            #if epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    self.model.save("checkpoint", epoch)
                #self.update_validation_accuracy(epoch)

            if epoch > 0 and epoch % 5 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2

        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()


    def run(self):

        self.model.eval()

        for i, data in enumerate(self.val_loader, 0):
            if data[1].size()[0] != self.model.batch_size:
                continue

            in_data = Variable(data[0].cuda())

            out_data = self.model(in_data)
            from IPython import embed; embed(); exit(-1)
            
            results_dir = os.path.join("run", self.model.name)
            print data[2]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            write_image_pc(os.path.join(results_dir, data[2][0]+"_mrt"),
                    (data[1][0, :, :, :], out_data[0, :, :].data.cpu()))
            print "Test PC saved."


    def evaluate(self):
        n_batches = 0
        all_correct_points = 0
        miou = 0

        total_error_class = np.zeros((13, 2))
        total_count_class = np.zeros(13)

        in_data = None
        out_data = None
        target = None

        n_iter = 0.0
        total_d = 0.0

        self.model.eval()

        for i, data in enumerate(self.val_loader, 0):
            if data[1].size()[0] != self.model.batch_size:
                continue

            in_data = Variable(data[0].cuda())
            target = Variable(data[1]).cuda()
            class_id = data[2][0]

            out_data = self.model(in_data)
            
            #indices = np.arange(4096)
            #np.random.shuffle(indices)
            #indices = indices[:1024]
            #out_data = out_data[:, :, indices]
            #target = target[:, :, indices]

            pd = Ops.batch_pairwise_dist(out_data.transpose(1,2), 
                target.transpose(1,2))
            pd = torch.sqrt(pd)
            #it_d = torch.min(pd, dim=2)[0].sum() + torch.min(pd, dim=1)[0].sum()
            total_error_class[class_id, 0] += torch.min(pd, dim=2)[0].data.cpu().numpy().mean()
            total_error_class[class_id, 1] += torch.min(pd, dim=1)[0].data.cpu().numpy().mean()
            total_count_class[class_id] += 1.0

            scalar_group = {}

            for c in xrange(13):
                if total_count_class[c] > 0.0:
                    scalar_group['class{}_error_pred'.format(c)] =  total_error_class[c, 0]/total_count_class[c]
                    scalar_group['class{}_error_gt'.format(c)] =  total_error_class[c, 1]/total_count_class[c]

            np.save('total_error_class.npy', total_error_class)
            np.save('total_count_class.npy', total_count_class)
            self.writer.add_scalars('class_errors', scalar_group, i)

            #total_d += it_d.data.cpu().numpy()

            n_iter += self.model.batch_size
            #self.writer.add_scalar('test_run', total_d/n_iter, i)

            if i < 50:
                results_dir = os.path.join("eval", self.model.name)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                write_image_pc(os.path.join(results_dir, "out_{}".format(str(2*i).zfill(4))),
                        (data[3][0, :, :, :], out_data[0, :, :].data.cpu()))
                save_torch_pc(os.path.join(results_dir, "out_{}.obj".format(str(2*i+1).zfill(4))), target)
                print "Test PC saved."

        np.save('total_error_class.npy', total_error_class)
        np.save('total_count_class.npy', total_count_class)
        print total_d/n_iter


class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.logger = DataVis.LossLogger(model.name)
        self.val_logger = DataVis.LossLogger("Validation-"+model.name)
        self.train_logger = DataVis.LossLogger("Training acc. "+model.name)

        self.model.cuda()


    def train(self, n_epochs, save_step=5):

        best_acc = 0

        for epoch in range(n_epochs):
            out_data = None
            in_data = None

            for i, data in enumerate(self.train_loader, 0):
                if data[1].size()[0] != self.model.batch_size:
                    continue

                in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                loss = self.loss_fn(out_data, target)

                self.logger.update(loss)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                self.train_logger.update(correct_points.float()/self.model.batch_size)

                loss.backward()
                self.optimizer.step()


            epoch_acc = self.update_validation_accuracy(epoch)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.model.save("checkpoint", epoch)
                #self.update_validation_accuracy(epoch)

            if epoch > 0 and epoch % 5 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2

    def update_validation_accuracy(self, epoch):
        n_batches = 0
        all_correct_points = 0
        miou = 0

        in_data = None
        out_data = None
        target = None

        results_dir = os.path.join("bad_results", self.model.name, 
            "epoch_{}".format(str(epoch).zfill(4)))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        wrong_results = []
        wrong_label = []
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)

        self.model.eval()

        for i, data in enumerate(self.val_loader, 0):
            if data[1].size()[0] != self.val_loader.batch_size:
                continue

            in_data = Variable(data[1], volatile=True).cuda()
            target = Variable(data[0], volatile=True).cuda().long()

            out_data = self.model(in_data)

            pred = torch.max(out_data, 1)[1]

            results = pred == target
            for i in xrange(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_results.append(in_data[i, :, :].cpu().data.numpy())
                    wrong_label.append(pred.cpu().data.numpy().astype('int')[i])
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            n_batches += 1

        #self.model.save_bad_results(wrong_results, wrong_label, results_dir)
        acc = all_correct_points.float() / (n_batches*self.val_loader.batch_size)
        self.val_logger.update(acc, step=1)

        print np.mean((samples_class-wrong_class)/samples_class)

        self.model.train()

        return acc.cpu().data.numpy()[0]


class VAETrainer(object):

    def __init__(self, model, loader, optimizer, loss_fn):

        self.optimizer = optimizer
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.logger = DataVis.LossLogger(model.name)
        self.reg_logger = DataVis.LossLogger("encreg-{}".format(model.name))

        self.model.cuda()


    def train(self, n_epochs, save_step=50):

        for epoch in range(n_epochs):
            out_data = None

            for i, data in enumerate(self.loader, 0):
                if data.size()[0] != self.model.batch_size:
                    continue

                in_data = Variable(data.cuda())

                self.optimizer.zero_grad()

                out_data = self.model(in_data)
                loss = self.loss_fn(out_data, in_data)
                reg = self.model.encoding_regularizer(in_data)
                print "Total loss: {} | Rec Loss: {} | KLD: {}".format(
                        loss.cpu().data.numpy()[0] + reg.cpu().data.numpy()[0],
                        loss.cpu().data.numpy()[0],
                        reg.cpu().data.numpy()[0])

                loss += reg
                
                self.logger.update(loss)
                self.reg_logger.update(reg)

                loss.backward()
                self.optimizer.step()

            if epoch % save_step == 0:
                self.model.save("checkpoint", epoch)

                results_dir = os.path.join("results", self.model.name, 
                    "epoch_{}".format(str(epoch).zfill(4)))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                samples_dir = os.path.join("samples", self.model.name, 
                    "epoch_{}".format(str(epoch).zfill(4)))
                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2

                try:
                    self.model.save_results(results_dir, out_data)
                    self.model.save_results(samples_dir, self.model.sample())
                except NotImplementedError:
                    print "Intermediate results not saved."


class AdversarialTrainer(object):
    
    def __init__(self, 
            d_model, 
            g_model,
            d_opt,
            g_opt,
            d_loss_fn,
            g_loss_fn,
            loader,
            encoder,
            decoder,
            batch_size=64):

        self.discriminator = d_model
        self.generator = g_model
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.loader = loader

        self.d_logger = DataVis.LossLogger(self.discriminator.name)
        self.dacc_real_logger = DataVis.LossLogger("ACCReal"+self.discriminator.name)
        self.dacc_fake_logger = DataVis.LossLogger("ACCFake_"+self.discriminator.name)
        self.g_logger = DataVis.LossLogger(self.generator.name)

        self.encoder = encoder
        self.decoder = decoder

        self.batch_size = batch_size

        self.noise = torch.FloatTensor(self.batch_size, self.generator.enc_size)

        self.discriminator.cuda()
        self.generator.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        
        self.encoder.eval()
        self.decoder.eval()


    def reset_grad(self):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    
    def train(self, n_epochs, save_step=5):

        for epoch in range(n_epochs):
            fake_data = None
            real_data = None

            for i, data in enumerate(self.loader, 0):
                if data.size()[0] != self.batch_size:
                    continue

                self.noise.normal_()
                real_data = self.encoder(Variable(data.cuda()))
                fake_data = self.generator(Variable(self.noise.cuda()))

                d_out_real = self.discriminator(real_data)
                d_out_fake = self.discriminator(fake_data)

                #Discriminator update
                self.d_opt.zero_grad()
                dloss_real = self.d_loss_fn((d_out_real, 
                    Variable(torch.ones(self.batch_size).cuda())))
                dloss_fake = self.d_loss_fn((d_out_fake, 
                    Variable(torch.zeros(self.batch_size).cuda())))

                dacc_real = torch.mean(F.sigmoid(d_out_real[0]))
                dacc_fake = 1 - torch.mean(F.sigmoid(d_out_fake[0]))
                self.dacc_real_logger.update(dacc_real)
                self.dacc_fake_logger.update(dacc_fake)
                dacc = 0.5*(dacc_real + dacc_fake)

                dloss = dloss_real + dloss_fake
                self.d_logger.update(dloss)

                self.reset_grad()
                dloss.backward(retain_variables=True)
                if dacc < 0.8:
                    self.d_opt.step()

                #Generator update
                self.g_opt.zero_grad()

                gloss = self.g_loss_fn((d_out_real, d_out_fake))
                #gloss = self.d_loss_fn((d_out_fake,
                #    Variable(torch.ones(self.batch_size).cuda())))
                self.g_logger.update(gloss)

                self.reset_grad()
                gloss.backward()
                self.g_opt.step()

            if epoch % save_step == 0:
                self.generator.save("checkpoint", epoch)
                self.discriminator.save("checkpoint", epoch)

                results_dir = os.path.join("results", self.generator.name, 
                    "epoch_{}".format(str(epoch).zfill(4)))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                try:
                    self.generator.save_results(results_dir, self.decoder(fake_data))
                except NotImplementedError:
                    print "Intermediate results not saved."


