import datetime

from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import *

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('Serif', 'LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Bold', 'LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Italic', 'LiberationSerif-Italic.ttf'))
pdfmetrics.registerFont(TTFont('Courier', './asset/Courier.ttf'))

from pathlib import Path
from pytorch_med_imaging.utils.visualization import draw_grid_contour
from torchvision.utils import make_grid
import imageio
import torch
import SimpleITK as sitk
import json

import numpy as np

def inch_to_pt(inch):
    return 72 * inch

def draw_image(img, seg, mode=0):
    r"""
    Draw based of mode. If `mode=0`, 3 slices surrounding the center of image will be drawn,
    if `mode=1`, 3 slices with the larges segmentation volume will be drawn
    Args:
        img:
        seg:

    Returns:

    """
    if isinstance(img, (str, Path)):
        img = sitk.ReadImage(img)
    if isinstance(seg, (str, Path)):
        seg = sitk.ReadImage(seg)

    if mode == 0:
        tissue_mask = sitk.HuangThreshold(img, 0, 1, 200)
        f = sitk.LabelShapeStatisticsImageFilter()
        f.Execute(tissue_mask == 1)
        cent = f.GetCentroid(1)
        cent = seg.TransformPhysicalPointToIndex(cent)

        # Compute square bounding box
        bbox = f.GetBoundingBox(1)
        w = bbox[3] - bbox[1]
        h = bbox[4] - bbox[2]
        l = max([w, h])

        img = sitk.GetArrayFromImage(img[bbox[0]:bbox[0] + l, bbox[1]:bbox[1] + l, bbox[2]:bbox[5]])
        slice_range = [cent[-1] - 1, cent[-1] + 1]
        return make_grid(torch.as_tensor(img[slice_range[0]:slice_range[1]+1]).unsqueeze(1), nrow=3, padding=2,
                         normalize=True).numpy().transpose(1, 2, 0)
    elif mode == 1:
        counts = np.sum(sitk.GetArrayFromImage(seg), axis=(1, 2))
        r = len(counts) - np.argsort(counts, kind='stable')  # reverse sort
        # check if there are more than 3 non-zero slices
        non_zero_slices = (counts != 0).sum()

        f = sitk.LabelShapeStatisticsImageFilter()
        f.Execute(seg >= 1)
        cent = f.GetCentroid(1)
        cent = seg.TransformPhysicalPointToIndex(cent)
        bbox = f.GetBoundingBox(1)
        w = bbox[3] - bbox[1]
        h = bbox[4] - bbox[2]
        l = max([w, h]) + 100    # fit to the size of the segmentation bounding box
        l = min([400, l])       # but no larger than 400 x 400

        crop = {
            'center': cent,
            'size': [l, l]
        }

        img, seg = (torch.as_tensor(sitk.GetArrayFromImage(x).astype('float')) for x in [img, seg])
        img, seg = img[r <= 3], seg[r <= 3]
        # Make black slices
        if non_zero_slices < 3:
            img[-non_zero_slices:] = 0
        return draw_grid_contour(img, [seg], color=[(255, 100, 55)],
                                 nrow=3, padding=2, thickness=1, crop=crop, alpha=0.8)

class LineSeparator(Flowable):
    """A horizontal line spanning the whole canvas"""
    def __init__(self, thickness=1, frame_padding=6):
       #these are hints to packers/frames as to how the floable should be positioned
        self.hAlign = 'CENTER'    #CENTER/CENTRE or RIGHT
        self.vAlign = 'MIDDLE'  #MIDDLE or TOP
        self.spaceBefore = 6
        self.spaceAfter = 6

        self.thickness = thickness # pt
        self.frame_padding = frame_padding

    def wrap(self, aW, aH):
        self.span = aW
        return (aW, self.thickness)

    def draw(self):
        canvas = self.canv
        p = canvas.beginPath()
        p.moveTo(-self.frame_padding, 0)
        p.lineTo(self.span + self.frame_padding, 0)
        p.close()
        canvas.setLineWidth(self.thickness)
        canvas.drawPath(p)

class ReportGen_NPC_Screening(Canvas):
    r"""
    Args:
        path (str or path):
            PDF file path.
    """ #
    def __init__(self, *args, **kwargs):
        super(ReportGen_NPC_Screening, self).__init__(*args, **kwargs)
        # self.saveState()
        self.setFont('Serif', 12)
        self.page_setting = {
            'size': A4,
            'width': A4[0],
            'height': A4[1],
            'margin': 36, # pt
            'padding': 6, # pt,
        }
        self.page_setting['frame_size'] = tuple(
            np.asarray(self.page_setting['size']) - self.page_setting['margin'] * 2
        )
        self.page_setting['frame_corner'] = tuple(
            [self.page_setting['margin'],
             self.page_setting['margin']] +
            list(np.asarray(self.page_setting['size']) - self.page_setting['margin'])
        )

        # Assets
        self.logo_text = Paragraph("""
                              <para align=center spaceb=3 face="times">
                              <b>The Chinese University of Hong Kong<br/>
                              Department of Imaging and Interventional Radiology</b><br/>
                              - <br/>
                              MRI nasopharyngeal carcinoma screening: <br/>
                              Artificial intelligent detection program
                              </para>
                              """)

        # Data
        self._dicom_tags = None


    def build_frames(self):
        # load images
        icon_height = 58
        icon = Image('./asset/Logo.jpg', height=icon_height, width=icon_height * 2)
        page_margin = self.page_setting['margin']
        width, height = self.page_setting['size']
        width_frame, height_frame = (width - 2 * page_margin)/2., inch_to_pt(1.9)

        # position at top left corner
        pos_x, pos_y = page_margin, self.page_setting['frame_corner'][3] - height_frame
        self.logo_frame = Frame(pos_x, pos_y, width_frame, height_frame,
                                leftPadding=6,
                                bottomPadding=6,
                                rightPadding=6,
                                topPadding=6,
                                showBoundary=1)
        self.logo_frame.addFromList([icon, self.logo_text], self)
        self.logo_frame.drawBoundary(self)

        # Patients particulars
        pos_x, pos_y = np.asarray(self.page_setting['frame_corner'][2:]) - np.asarray([width_frame, height_frame])
        self.patient_frame = Frame(pos_x, pos_y, width_frame, height_frame,
                                leftPadding=6, bottomPadding=6, rightPadding = 6, topPadding = 6,
                                showBoundary=1)
        self.patient_frame.drawBoundary(self)

        # text description container
        self.frame = Frame(page_margin, page_margin,
                           self.page_setting['frame_size'][0],
                           self.page_setting['frame_size'][1] - height_frame,
                           leftPadding=self.page_setting['padding'],
                           bottomPadding=self.page_setting['padding'],
                           rightPadding=self.page_setting['padding'],
                           topPadding=self.page_setting['padding'],
                           showBoundary=1)
        self.frame.drawBoundary(self)

    def enrich_patient_data(self):
        items_display = {
            'Name': None,
            'Chinese Name': None,
            'Sex': None,
            'Age': None,
            'Scan date': None,
            'Patient ID': None,
            'Scanner': None,
            'Protocol': None,
            'Report gen. date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        # update the json
        self._dicom_tags = {k: v for k, v in self._dicom_tags.items() if v is not None}
        items_display.update(self._dicom_tags)
        print(self._dicom_tags, items_display)

        # msg = '<br/>'.join([f"{key}: \t<u>{value}</u>" for key, value in items_display.items()])
        pretag, posttag = "<para spaceBefore=2 fontSize=8>", "</para>"
        pars = [Paragraph(pretag + f"{key}: \t<u>{value}</u>" + posttag) for key, value in items_display.items()]
        # text = Paragraph(f"<para spaceBefore=3 spaceAfter=6> {msg} </para>")
        self.patient_frame.addFromList(pars, self)

    def enrich_mri_frame(self):
        data_display = {
            'image_dir': './example_data/npc_case/case.png', #TODO: Change this to dynamic load
            'diagnosis_radiomics': None,
            'diagnosis_dl': None,
            'diganosis_overall': None,  # {0: healthy/benign hyperplasia, 1: carcinoma, -1: doubt}
            'operator_remarks': None,
            'ref_radiomics': None,
            'ref_dl': None,
            'lesion_vol': None,  # Unit is cm3
            'remark': None
        }

        column_map = {
            'diagnosis_dl': "Deep learning prediction",
            'diagnosis_radiomics': "Radiomics prediction",
            'ref_radiomics': "Reference for normal",
            "ref_dl": "Reference for normal",
            'lesion_vol': "Lesion volume"
        }
        # Load data
        # TODO: write function to load properties from data

        story = []
        fixed_aspect_ratio = 3
        im = Image(data_display['image_dir'],
                   width  = 1E10,
                   height = (self.page_setting['frame_size'][0] - self.page_setting['padding'] * 2) // 3,
                   kind='proportional')

        # If healthy case, display COM near the nasopharynx
        msg = "No abnormally found (Three slices near the nasopharynx displayed)"
        color = "#469e54"

        # If benign case or NPC case, display segmentation
        msg = "Lesion detected (Three or less slices with largest tumor volume displayed)"
        color = "#a32e2c"

        # If doubtful case, display segmentation with red warning sign
        style = getSampleStyleSheet()['Heading2']
        sect_title = Paragraph(f"<para face=times color={color}><b>" + msg + "</b></para>", style=style)
        story.extend([sect_title, im])

        # Draw a horizontal line for seperation
        story.append(LineSeparator(1, self.page_setting['padding']))

        # build description
        style = getSampleStyleSheet()['Heading2']
        desc_title = Paragraph(f"<para face=times spaceBefore=0> <b><u> Description </u></b></para>", style=style)
        story.append(desc_title)

        # lesion properties
        prop = []
        for val in ['lesion_vol']:
            msg = f"<para face=courier fontSize=11 spaceAfter=10>>{column_map[val]} -- <u>{data_display[val]}</u>(cm^3)</para>"
            prop.append(Paragraph(msg))
        story.extend(prop)

        desc = []
        for val, ref in zip(['diagnosis_radiomics', 'diagnosis_dl'],
                            ['ref_radiomics', 'ref_dl']):
            msg =  f"<para face=courier fontSize=11 spaceAfter=10>>{column_map[val]} -- <u>{data_display[val]}</u><br/></para>"
            desc.append(Paragraph(msg))
            msg =  f"<para face=courier fontSize=11 leftIndent=24 spaceAfter=10>>{column_map[ref]} -- {data_display[ref]}<br/></para>"
            desc.append(Paragraph(msg))

        story.extend(desc)
        self.frame.addFromList(story, self)
        pass

    def prepare_canvas(self):
        r"""
        Draw the logos and boxes
        """
        page_margin = inch_to_pt(0.5) # pt
        frame_padding = inch_to_pt(0.1) # pt
        width, height = self.page_size
        width_frame, height_frame = width - 2 * page_margin, height - 2 * page_margin

        story = []
        self.frame = Frame(page_margin, page_margin, width_frame, height_frame,
                           leftPadding=6, bottomPadding=6, rightPadding = 6, topPadding = 6,
                           showBoundary=1)
        self.frame.addFromList([BalancedColumns([self.logo_text, Paragraph("Randomtext")])], self)
        self.frame.drawBoundary(self)

    def draw(self):
        self.build_frames()
        self.enrich_patient_data()
        self.enrich_mri_frame()
        self.showPage()
        self.save()


    def _read_details_from_dicom(self, dicom_dir, sequence_id=None):
        pass

    def _read_details_from_dicom_json(self, json_file):
        dicom_data = json.load(open(str(json_file), 'r'))
        dicom_data_tags = {
            'Name': '0010|0010',
            'Chinese Name': None,
            'Sex': '0010|0040',
            'Age': '0010|1010',
            'Scan date': '0008|0020',
            'Patient ID': '0010|0020',
            'Scanner': '0008|1010',
            'Protocol': '0018|1030',
            'Report gen. date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        out = {v: dicom_data.get(dicom_key, None) for v, dicom_key in dicom_data_tags.items()}
        return out

    @property
    def dicom_tags(self):
        return self._dicom_tags

    @dicom_tags.setter
    def dicom_tags(self, val):
        if isinstance(val, (str, Path)):
            d = self._read_details_from_dicom_json(val)
        self._dicom_tags = d


if __name__ == '__main__':
    import pprint
    im = draw_image('./example_data/npc_case/1280-T2_FS_TRA+301.nii.gz',
                    './example_data/npc_case/1280.nii.gz', mode=1)
    # print(im.shape, type(im))
    imageio.imsave('./example_data/npc_case/case.png', im)
    #
    c = ReportGen_NPC_Screening("hello.pdf")
    c.dicom_tags = './example_data/npc_case/1280_dicom_tag.json'
    pprint.pprint(c.page_setting)
    c.draw()