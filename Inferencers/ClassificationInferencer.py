from .InferencerBase import InferencerBase
from MedImgDataset import ImageDataSet, DataLabel
from torch.utils.data import DataLoader
from tqdm import *
from torch.autograd import Variable
import os
import torch
import torch.nn.functional as F
import numpy as np
from SimpleITK import WriteImage, ReadImage, GetImageFromArray
from Networks.GradCAM import *
from torchvision.utils import make_grid
from matplotlib.pyplot import imsave
import cv2
from Algorithms.visualization import draw_overlay_heatmap


class ClassificationInferencer(InferencerBase):
    def __init__(self, input_data, out_dir, batch_size, net, checkpoint_dir, iscuda, logger):
        inference_configs = {}
        inference_configs['indataset']      = input_data
        inference_configs['batchsize']      = batch_size
        inference_configs['net']            = net
        inference_configs['netstatedict']   = checkpoint_dir
        inference_configs['logger']         = logger
        inference_configs['outdir']         = out_dir
        inference_configs['iscuda']         = iscuda

        super(ClassificationInferencer, self).__init__(inference_configs)

    def _input_check(self):
        assert isinstance(self._in_dataset, ImageDataSet), "Type is %s"%(type(self._in_dataset))
        return 0

    def _create_net(self):
        in_chan = self._in_dataset[0].size()[0]
        out_chan = 2 #TODO: Temp fix

        try:
            self._net = self._net(in_chan, out_chan, save_mask=False, save_weight=True)
            self._ATTENTION_FLAG=True
        except:
            self._logger.log_print_tqdm("Cannot create network with 'save_mask' attribute!", 20)
            self._net = self._net(in_chan, out_chan)
            self._ATTENTION_FLAG=False

        self._logger.log_print_tqdm("Loading checkpoint from: " + self._net_state_dict, 20)
        self._net.load_state_dict(torch.load(self._net_state_dict), strict=False)
        # self._net = nn.DataParallel(self._net)
        self._net.train(False)
        self._net.eval()
        if self._iscuda:
            self._net = self._net.cuda()


        return self._net

    def _create_dataloader(self):
        self._data_loader = DataLoader(self._in_dataset, batch_size=self._batchsize,
                                       shuffle=False, num_workers=0, drop_last=False)
        return self._data_loader

    def attention_write_out(self, attention_list):
        attention_outdir = os.path.dirname(self._outdir)
        id_lists = self._in_dataset.get_unique_IDs()
        temp_atten_list = [t for t in zip(*attention_list)]
        assert len(id_lists) == len(temp_atten_list), \
            "Length mismatch! %s vs %s"%(len(id_lists),len(temp_atten_list))

        for i, (id, atten) in enumerate(tqdm(zip(id_lists, temp_atten_list))):
            # obtain meta information of the data
            metadata = self._in_dataset.metadata[i]
            ref_im = ReadImage(self._in_dataset.get_data_source(i))
            ori_size = np.array(ref_im.GetSize())
            ori_spacing = np.array(ref_im.GetSpacing())
            for j in range(len(atten)):
                # calculate new spacing based on size
                atten_im = atten[j].numpy().transpose([2, 3, 1, 0])
                atten_size = atten_im.shape[:3]
                ratio = ori_size / atten_size
                new_space = ori_spacing * ratio

                write_out_image = GetImageFromArray(atten_im.transpose([2,0,1,3]))
                write_out_image.CopyInformation(ref_im)
                # write_out_image.SetOrigin(ref_im.GetOrigin())
                # write_out_image.SetDirection(ref_im.GetDirection())
                write_out_image.SetSpacing(new_space)

                #
                # for k in [1, 2, 3]:
                #     new_metadata['pixdim[%s]'%k] = "%.05f"%new_space[k-1]
                #
                # write_out_image = ImageDataSet.WrapImageWithMetaData(
                #     atten_im.transpose([2, 0, 1, 3]), new_metadata)
                WriteImage(write_out_image, os.path.join(attention_outdir,
                                                         str(id) + '_attention_%02d.nii.gz'%j))

        pass


    def grad_cam_write_out(self, target_layer):
        gradcam = GradCam(self._net, target_layer)

        out_tensor = []
        cam_tensor = []
        last_batch_dim = 0
        for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
            s = samples
            if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                s = [Variable(ss, requires_grad=True).float() for ss in s]

            if self._data_loader:
                s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

            torch.no_grad()
            if isinstance(s, list):
                out, dec, cam = gradcam(*s)
            else:
                out, dec, cam = gradcam(s)

            while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                out = out.unsqueeze(0)
                self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

            out_tensor.append(out)
            cam_tensor.append(cam)
            last_batch_dim = out.dim()
            del out, s


        out_tensor = torch.cat(out_tensor, dim=0)
        cam_tensor = torch.cat(cam_tensor, dim=0)

        dl = self._writter(out_tensor)
        print(dl._data_table.to_string())


        ids = self._in_dataset.get_unique_IDs()
        outdir = os.path.dirname(self._outdir)
        for i in tqdm(range(len(self._in_dataset))):
            t, c = self._in_dataset[i], cam_tensor[i].squeeze()

            # normalize slice by slice to range 0-1
            for j, slice in enumerate(c):
                _tmp = c[j]
                if not _tmp.max() == 0:
                    _tmp = _tmp - float(_tmp.min())
                    _tmp = _tmp / float(_tmp.max())
                    c[j] = _tmp
            t_grid = make_grid(t.squeeze().unsqueeze(1), nrow=5, padding=1, normalize=True)
            c_grid = make_grid(c.squeeze().unsqueeze(1), nrow=5, padding=1, normalize=True)

            hm = draw_overlay_heatmap(t_grid, c_grid)
            outname = os.path.join(outdir, "%s_gradcam.jpg"%ids[i])
            imsave(outname, hm)



    def write_out(self):
        out_tensor = []
        out_decisions = {}
        out_attention = []
        out_slice_attention = []
        last_batch_dim = 0
        with torch.no_grad():
            for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
                s = samples
                if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                    s = [Variable(ss, requires_grad=False).float() for ss in s]

                if self._data_loader:
                    s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

                if isinstance(s, list):
                    out = self._net.forward(*s).squeeze()
                else:
                    out = self._net.forward(s).squeeze()

                while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

                out_tensor.append(out.data.cpu())
                if self._ATTENTION_FLAG:
                    # out_attention.append(self._net.get_mask())
                    out_slice_attention.append(self._net.get_slice_attention())

                last_batch_dim = out.dim()
                del out, s

            if self._ATTENTION_FLAG:
                # out_attention = [a for a in zip(*out_attention)]
                # out_attention = [torch.cat(a, dim=0) for a in out_attention]
                out_slice_attention = torch.cat(out_slice_attention)
                print(out_slice_attention.shape)

            out_tensor = torch.cat(out_tensor, dim=0)
            dl = self._writter(out_tensor, out_attention=out_attention, out_slice_attention=out_slice_attention)
            print(dl._data_table.to_string())

    def _writter(self, out_tensor, out_attention =None, out_slice_attention=None):
        out_decisions = {}
        out_decision = torch.argmax(out_tensor, dim=1)
        out_tensor = F.softmax(out_tensor, dim=1)
        if self._ATTENTION_FLAG and not out_attention is None:
            # self.attention_write_out(out_attention)
            pass
        if os.path.isdir(self._outdir):
            self._outdir = os.path.join(self._outdir, 'class_inf.csv')
        if not self._outdir.endswith('.csv'):
            self._outdir += '.csv'
        if os.path.isfile(self._outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self._outdir, 30)
        if not os.path.isdir(os.path.dirname(self._outdir)):
            os.makedirs(os.path.dirname(self._outdir), exist_ok=True)

        # Write decision
        out_decisions['IDs'] = self._in_dataset.get_unique_IDs()
        for i in range(out_tensor.shape[1]):
            out_decisions['Prob_Class_%s'%i] = out_tensor[:, i].data.cpu().tolist()
            if not out_slice_attention is None:
                for j in range(len(out_slice_attention[0])):
                    out_decisions['SA %02d'%j] = out_slice_attention[:,j]

        out_decisions['Decision'] = out_decision.tolist()
        dl = DataLabel.from_dict(out_decisions)
        dl.write(self._outdir)
        return dl