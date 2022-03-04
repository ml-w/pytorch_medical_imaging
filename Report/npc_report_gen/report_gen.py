import datetime
import tempfile

import reportlab.pdfgen.canvas
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import *

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('Serif', 'LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Bold', 'LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Serif-Italic', 'LiberationSerif-Italic.ttf'))
# pdfmetrics.registerFont(TTFont('Courier', './asset/Courier.ttf'))

from pathlib import Path
from pytorch_med_imaging.utils.visualization import draw_grid_contour
from pytorch_med_imaging.logger import Logger
from torchvision.utils import make_grid
from typing import Union, Callable, Optional, Iterable
import imageio
import torch
import SimpleITK as sitk
import json
import imageio
import re

import numpy as np

def inch_to_pt(inch):
    return 72 * inch



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

class InteractiveCheckBox(Flowable):
    def __init__(self, text='A Box'):
        Flowable.__init__(self)
        self.text = text
        self.boxsize = 6

    def draw(self):
        self.canv.saveState()
        self.canv.translate(len(self.text) * self.boxsize + 2, 0)
        self.canv.drawString(-len(self.text) * self.boxsize - 2, 0, self.text)
        form = self.canv.acroForm
        form.checkbox(checked=False,
                      buttonStyle='check',
                      name=self.text,
                      tooltip=self.text,
                      relative=True,
                      size=self.boxsize)
        self.canv.restoreState()
        return

    # def wrap(self, aW, aH):
    #     aW = self.boxsize + len(self.text) * self.boxsize
    #     aH = self.boxsize
    #     return (aW, aH)


class ReportGen_NPC_Screening(Canvas):
    r"""
    Args:
        path (str or path):
            PDF file path.
    """ #
    def __init__(self, *args, **kwargs):
        super(ReportGen_NPC_Screening, self).__init__(*args, **kwargs)
        # self.saveState()
        #==========
        # Settings
        #==========
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
        self.image_setting = {
            'max_slice_size': 400,  # Max w, h of the displayed slices (px)
            'padding_size': 150,    # Number of pixels to pad around the tumor. If 0, dispaly tumor bounding box as FOV
            'nrow': 3,              # Number of slices per row, pass to `make_grid
            'n': 3,                 # Max number of slices to display, slices with largest volumes are displayed.
        }

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
        self._data_root_path = None
        self.diagnosis_overall = None

        # Set up logger
        self._logger = Logger['ReportGen']


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
        frame_size_ = self.page_setting['frame_size']
        frame_padding_ = self.page_setting['padding']
        self.frame = Frame(page_margin, page_margin,
                           frame_size_[0],
                           frame_size_[1] - height_frame,
                           leftPadding=frame_padding_,
                           bottomPadding=frame_padding_,
                           rightPadding=frame_padding_,
                           topPadding=frame_padding_,
                           showBoundary=1)
        self.frame.drawBoundary(self)

        self.frame_p2 = Frame(page_margin, page_margin,
                              frame_size_[0],
                              frame_size_[1],
                              leftPadding=frame_padding_,
                              bottomPadding=frame_padding_,
                              rightPadding=frame_padding_,
                              topPadding=frame_padding_,
                              showBoundary=1)

        self.table_frame = Frame(page_margin + frame_size_[0] // 2,
                                 page_margin,
                                 frame_size_[0] // 2,
                                 frame_size_[1],
                                 leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)

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
            'image_dir': None,
            'diagnosis_radiomics': None,
            'diagnosis_dl': None,
            'diganosis_overall': None,  # {0: healthy/benign hyperplasia, 1: carcinoma, -1: doubt}
            'operator_remarks': None,
            'ref_radiomics': None,
            'ref_dl': None,      # larger => NPC
            'lesion_vol': None,  # Unit is cm3
            'remark': "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                      "Pellentesque ut metus et eros placerat lobortis at sed leo. "
                      "Quisque a semper arcu. Morbi et aliquam sem. Nulla rutrum "
                      "varius nibh non sollicitudin. ",
            "user_grade": u"1 \xe2 \t 2 \xe2 \t 3 \xe2"
        }
        # Load data
        data_display.update(self.data_display)
        self._logger.debug(f"data_display: {data_display}")

        column_map = {
            'diagnosis_dl': "Deep learning prediction",
            'diagnosis_radiomics': "Radiomics prediction",
            'ref_radiomics': "Reference for normal",
            "ref_dl": "Reference for normal",
            'lesion_vol': "Lesion volume",
            'remark': 'Comment',
            'user_grade': 'Grading'
        }
        _ref_dl = re.search("[0-9\.\-]+", str(data_display['ref_dl'])).group()
        data_display['ref_dl'] = '< ' + str(_ref_dl)
        diagnosis_overall = self.get_overall_diagnosis(data_display['lesion_vol'],
                                                       data_display['diagnosis_dl'],
                                                       data_display['diagnosis_radiomics'],
                                                       lambda x: x >= _ref_dl,
                                                       lambda x: x >= data_display['ref_radiomics'])
        self.diagnosis_overall = diagnosis_overall
        # draw & save image
        im, sn = self.draw_image(data_display['image_nii'], data_display['segment_nii'], return_num_seg=True)
        im_file_ = tempfile.NamedTemporaryFile(mode='wb', suffix='.png')
        imageio.imsave(im_file_, im, format='png')
        data_display['image_dir'] = im_file_.name

        self._logger.debug(f"diagnosis: {diagnosis_overall}")

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
        if diagnosis_overall in [-1, 1]:
            _d = "Doubtful" if diagnosis_overall == -1 else "NPC"
            recommend = f"{_d}; recommend full scan"
            msg = f"Lesion detected (Three or less slices [{sn[0]} in {sn[1]} segmented] with largest tumor volume " \
                  f"displayed)"
            color = "#a32e2c"
        elif diagnosis_overall == 2:
            recommend = "Benign lesion, no further action needed"
            msg = f"Benign lesion detected (Three slices [{sn[0]} in {sn[1]} segmented] with largest lesion volume " \
                  f"displayed) "
            color = "#97a128"
        else:
            recommend = "No further action needed"
            msg = "No significant lesion detected (Three slices in the middle of image displayed)"
            color = "#32a852"

        # If doubtful case, display segmentation with red warning sign
        style = getSampleStyleSheet()['Heading3']
        sect_title = Paragraph(f"<para face=times color={color}><b>" + msg + "</b></para>", style=style)
        story.extend([sect_title, im])

        # Draw a horizontal line for seperation
        story.append(LineSeparator(1, self.page_setting['padding']))

        # build description
        style = getSampleStyleSheet()['Heading3']
        desc_title = Paragraph(f"<para face=times spaceBefore=0> <b><u> Description </u></b></para>", style=style)
        story.append(desc_title)

        # lesion properties
        prop = []
        for val in ['lesion_vol']:
            # msg = f"<para face=courier fontSize=11 spaceAfter=10>>{column_map[val]} -- <u>{data_display[val]}</u>(cm^3)</para>"
            msg = self.generate_key_value_msg(column_map[val], data_display[val], unit="(cm^3)")
            prop.append(Paragraph(msg))
        story.extend(prop)

        desc = []
        for val, ref in zip(['diagnosis_radiomics', 'diagnosis_dl'],
                            ['ref_radiomics', 'ref_dl']):
            msg = self.generate_key_value_msg(column_map[val], data_display[val], underline_value=True)
            desc.append(Paragraph(msg))
            msg = self.generate_key_value_msg(column_map[ref], data_display[ref], left_indent_level=1, underline_value=False)
            # msg =  f"<para face=courier fontSize=11 leftIndent=24 spaceAfter=10>>{column_map[ref]} -- {data_display[ref]}<br/></para>"
            desc.append(Paragraph(msg))
        story.extend(desc)

        # recommendation
        msg = self.generate_key_value_msg("Overall diagnosis", recommend, value_tags=f"color={color}",
                                          underline_value=True, bold_value=True)
        story.append(Paragraph(msg))

        # separator
        story.append(LineSeparator(1, self.page_setting['padding']))
        right_indent = int(self.page_setting['frame_size'][0] // 2 + 10) # Room for the table
        style = getSampleStyleSheet()['Heading3']
        desc_title = Paragraph(f"<para face=times spaceBefore=0> <b><u> Clinical Impression </u></b></para>", style=style)
        story.append(desc_title)

        # For user grading
        msg = f"<para face=courier fontSize=11 spaceAfter=10 rightIndent={right_indent}>>{column_map['user_grade']}</para>"
        story.append(Paragraph(msg))
        grade_table = Table([[InteractiveCheckBox(text=f"{i}") for i in list(range(1, 5)) + ['5a', '5b']]],
                        colWidths=25)
        grade_table.hAlign="LEFT"
        story.extend([grade_table])

        # remark
        msg = f"<para face=courier fontSize=11 spaceAfter=10 rightIndent={right_indent}>>" \
              f"{column_map['remark']}: <br/></para>"
        story.append(Paragraph(msg))
        msg = f"<para face=courier fontSize=11 spaceAfter=10 rightIndent={right_indent } leftIndent=12>" \
              f"{data_display['remark']}</para>"
        story.append(Paragraph(msg))

        self.frame.addFromList(story, self)
        self.table_frame.addFromList([self.draw_grading_table()], self)
        im_file_.close()

        # New page displaying all segmented slides, max num of slide displayed without messing layout is 20
        self.showPage()
        im = self.draw_image(data_display['image_nii'], data_display['segment_nii'], return_num_seg=True, mode=2)
        im_file_ = tempfile.NamedTemporaryFile(mode='wb', suffix='.png')
        imageio.imsave(im_file_, im, format='png')
        data_display['image_dir'] = im_file_.name

        story = []
        style = getSampleStyleSheet()['Heading2']
        msg = f"Grid of all segmented slices (PID: {self._dicom_tags['Patient ID']})"
        sect_title = Paragraph(f"<para face=times align=center><b>" + msg + "</b></para>", style=style)

        fixed_aspect_ratio = 3
        im = Image(data_display['image_dir'],
                   width  = (self.page_setting['frame_size'][0] - self.page_setting['padding'] * 2),
                   height = 1E10,
                   kind='proportional')
        story.extend([sect_title,
                      LineSeparator(1, self.page_setting['padding']),
                      im,
                      LineSeparator(1, self.page_setting['padding'])])

        self.frame_p2.drawBoundary(self)
        self.frame_p2.addFromList(story, self)
        im_file_.close()
        pass

    @staticmethod
    def generate_key_value_msg(key,
                               value,
                               *added_tags,
                               unit: Optional[str] = "",
                               label: Optional[str] = " -- ",
                               font_size: Optional[int] = 11,
                               left_indent_level: Optional[int] = 0,
                               right_indent: Optional[int] = 0,
                               underline_value: Optional[bool] = True,
                               bold_value: Optional[bool] = False,
                               value_tags: Optional[Iterable[str]] = None,
                               ):

        left_indent = 2 + left_indent_level * (font_size + 3)
        msg = f"<para face=courier fontSize={font_size} leftIndent={left_indent} spaceAfter=10 " \
              f"rightIndent={right_indent} {' '.join(added_tags)}>"
        if underline_value:
            value = f"<u>{value}</u>"
        if bold_value:
            value = f"<b>{value}</b>"

        if value_tags is not None:
            value_tags = [value_tags] if isinstance(value_tags, str) else value_tags
            value = f"</para><para face=courier fontSize={font_size} spaceAfter=10 {' '.join(value_tags)}>{value}"

        msg += f">{key}{label}{value}{unit}<br/>"
        msg += "</para>"
        return msg

    def draw_grading_table(self):
        # data = [['<b>Grade</b>', '<b>Walls</b>', '<b>Adenoid</b>'],
        #         ['1: normal', '- Thin wall (1-3mm)', '- Absent/vestigial tags/nubbin'],
        #         ['2: probably benign hyperplasia',
        #          '- Diffuse thickening (>3mm), symmetric size, signal intensity and contour',
        #          '<u>CE scan</u>: <br/>'
        #          '- composed of Thornwaldt <br/>'
        #          '- cyst/multiple cysts, OR symmetric size, signal intensity, and contour <br/>'
        #          '- with preserved symmetric contrast-enhancing septa perpendicular to the roof, separated by less '
        #          'enhancing columns (ie, stripped appearance)<br/>'
        #          '<u>Plain scan</u>: <br/>'
        #          '- composed of Thornwaldt <br/>'
        #          '- cyst/multiple cysts'],
        #         ['3: indeterminate', '- Diffuse thickening (>3mm)<br/> '
        #                              '- asymmetric size or signal intensity or contour, which'
        #                              ' is non-expansile',
        #          '<u>CE scan</u>: <br/>'
        #          '- asymmetric size, signal intensity OR contour with preserved or partial disruption/internal '
        #          'distortion of contrast-enhancing septa<br/>'
        #          '<u>Plain scan</u>: <br/>'
        #          '- symmetric size, signal intensity, and contour'],
        #         ['4: suspicious for NPC',
        #          '- Diffuse thickening (>3mm) <br/>'
        #          '- asymmetric size or signal intensity or contour, which'
        #                              ' is expansile',
        #          '<u>CE scan</u>: <br/>'
        #          '- absent CE septa in focal adenoid, OR external distortion of CE septa by an adjacent roof mass '
        #          '<br/>'
        #          '<u>Plain scan</u>: <br/>'
        #          '- asymmetric size, signal intensity or contour'],
        #         ['5: probably NPC',
        #          '- focal mass',
        #          '<u>CE scan</u>: <br/>'
        #          '- absent CE septa in focal adenoid, OR external distortion of CE septa by an adjacent roof mass '
        #          'on at least 1 section'],
        #         ['5b', '- Spread outside of the nasopharynx (superficial or deep)'],
        #         ['5c', '- Metastic retropharyngeal or upper cervical nodes.']
        #         ]
        data = [['Grading system reference (King AD et al., 2020)'],
                ['<b>Grade</b>', '<b>Walls</b>', '<b>Adenoid</b>'],
                ['1: normal', '- thin wall (1-3mm)', '- absent/vestigial tags/nubbin'],
                ['2: probably benign hyperplasia',
                 '- diffuse thickening (>3mm), symmetric size, signal intensity and contour',
                 '- composed of Thornwaldt <br/>'
                 '- cyst/multiple cysts'],
                ['3: indeterminate', '- diffuse thickening (>3mm)<br/> '
                                     '- asymmetric size or signal intensity or contour, which'
                                     ' is non-expansile',
                 '- symmetric size, signal intensity, and contour'],
                ['4: suspicious for NPC',
                 '- diffuse thickening (>3mm) <br/>'
                 '- asymmetric size or signal intensity or contour, which'
                                     ' is expansile',
                 '- asymmetric size, signal intensity or contour'],
                ['5: probably NPC',
                 '- focal mass',
                 'no grade for plain scan'],
                ['5b', '- spread outside of the nasopharynx (superficial or deep)'],
                ['5c', '- metastic retropharyngeal or upper cervical nodes.']
                ]

        styles = getSampleStyleSheet()
        styleB = styles['BodyText']
        styleB.spaceBefore = 0
        styleB.spaceAfter = 4
        styleB.fontSize = 8
        styleB.leading = styleB.fontSize + 2
        styleB.leftIndent = 0
        data = [[Paragraph(s, styleB) for s in row] for row in data]
        data[0][0] = Paragraph('<i>Table: Grading system reference (King AD et al., 2020)</i>',
                               style=ParagraphStyle(name='_',
                                                    fontName='times',
                                                    alignment=1,
                                                    leading=15

                                                    ))

        print(data[0][0])
        table_style = TableStyle(
            [('LINEABOVE', (0, 1), (-1, 1), 1, "#000000"),
             ('LINEABOVE', (0, 2), (-1, 2), 0.5, "#000000"),
             ('LINEBELOW', (0, -1), (-1, -1), 1, "#000000"),
             ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
             ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
             ('TOPPADDING', (0, 1), (-1, -1), 1),
             ('ROWBACKGROUNDS', (0, 1), (-1, -1), ["#FFFFFF", "#ede9cc"]),
             ('VALIGN',(0,1),(-1,-1),'TOP'),
             ('ALIGN',(0,0),(-1,0),'CENTER'),
             ('SPAN', (1, -2), (-1, -2)),
             ('SPAN', (1, -1), (-1, -1)),
             ('SPAN', (0, 0), (-1, 0))]
        )
        t = Table(data, style=table_style, colWidths=[inch_to_pt(0.8), inch_to_pt(1.5), inch_to_pt(1.5)])
        t = TopPadder(t)
        t.hAlign = 'RIGHT'
        return t

    def draw(self):
        self.build_frames()
        self.enrich_patient_data()
        self.enrich_mri_frame()
        self.showPage()
        self.save()
        self._logger.info(f"Write PDF to: {self._filename}")

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

    def read_data_display(self, data_display: dict):
        r"""
        Wrapper function.
        Now the image is hard-coded to be "npc_report.png" under `self._data_root_path`
        """
        self.dicom_tags = data_display.pop('dicom_tags')
        self.data_display = data_display

    def draw_image(self, img, seg, mode=None, return_num_seg=False):
        r"""
        Draw based of mode. mode: 0 => Display center slices, 1 => Display with segmentation

        """
        if isinstance(img, (str, Path)):
            img = sitk.ReadImage(img)
        if isinstance(seg, (str, Path)):
            seg = sitk.ReadImage(seg)
        elif seg is None:
            mode = 0

        if mode is None:
            mode = int(not self.diagnosis_overall  == 0)

        if mode == 0:
            tissue_mask = sitk.HuangThreshold(img, 0, 1, 200)
            f = sitk.LabelShapeStatisticsImageFilter()
            f.Execute(tissue_mask == 1)
            cent = f.GetCentroid(1)
            cent = seg.TransformPhysicalPointToIndex(cent) # use seg instead of img because its faster

            # Compute square bounding box  [xstart, ystart, start, xsize, ysize, zsize]
            bbox = f.GetBoundingBox(1)
            w = bbox[3]
            h = bbox[4]
            l = max([w, h])

            n = self.image_setting['n']
            img = sitk.GetArrayFromImage(img[bbox[0]:bbox[0] + l, bbox[1]:bbox[1] + l, bbox[2]:bbox[5]])
            slice_range = [cent[-1] - n // 2, cent[-1] + n // 2]
            self.image = make_grid(torch.as_tensor(img[slice_range[0]:slice_range[1]+1]).unsqueeze(1),
                                   nrow=self.image_setting['nrow'], padding=2, normalize=True).numpy().transpose(1, 2, 0)
            if return_num_seg:
                return self.image, None
            else:
                return self.image
        elif mode == 1:
            counts = np.sum(sitk.GetArrayFromImage(seg), axis=(1, 2)).astype('int32')
            non_zero_indices = np.nonzero(counts)[0]    # argsort is broken by zeros, so extract non-zero elements first
            r = non_zero_indices[np.argsort(-counts[non_zero_indices])] # argsort element, then get its indices

            # check if there are more than 3 non-zero slices
            non_zero_slices = (counts != 0).sum()
            n = self.image_setting['n']

            f = sitk.LabelShapeStatisticsImageFilter()
            f.Execute(seg >= 1)
            cent = f.GetCentroid(1)
            cent = seg.TransformPhysicalPointToIndex(cent)
            bbox = f.GetBoundingBox(1)
            self._logger.info(f"BBox: {bbox}")
            w = bbox[3]
            h = bbox[4]
            l = max([w, h]) + self.image_setting['padding_size']      # fit to the size of the segmentation bounding box
            l = min([self.image_setting['max_slice_size'], l])        # but no larger than 'max_slice_size'

            crop = {
                'center': [cent[1], cent[0]], # y, x because of make_grid convention
                'size': [l, l]
            }

            display_slice = r[:3] # Get the three slices with largest tumor
            display_slice.sort()  # put them in axial order
            ori_size = seg.GetSize()
            img, seg = (torch.as_tensor(sitk.GetArrayFromImage(x).astype('float')) for x in [img, seg])
            img, seg = img[display_slice], seg[display_slice]

            # If not enough slices to display, make black slices
            if non_zero_slices < n:
                _s = img.shape[1:]
                _b = torch.zeros([n - non_zero_slices, _s[0], _s[1]])
                img = torch.cat([img, _b])
                seg = torch.cat([seg, _b])

            self.image = draw_grid_contour(img, [seg], color=[(255, 100, 55)],
                                           nrow=self.image_setting['nrow'], padding=2, thickness=1, crop=crop, alpha=0.8)

            if return_num_seg:
                num_seg = bbox[-1]
                all_slice = int(ori_size[-1])
                return self.image, (num_seg, all_slice)
            else:
                return self.image
        elif mode == 2: # Draw all slice
            ori_size = seg.GetSize()
            f = sitk.LabelShapeStatisticsImageFilter()
            f.Execute(seg >= 1)
            bbox = f.GetBoundingBox(1)
            cent = f.GetCentroid(1)
            cent = seg.TransformPhysicalPointToIndex(cent) # use seg instead of img because its faster

            # Compute square bounding box
            w = bbox[3]
            h = bbox[4]
            display_slide_u = min([bbox[2] + bbox[5] + 1, ori_size[-1] - 1])
            display_slide_d = max([bbox[2] - 1, 0])
            l = max([w, h])
            l = max([w, h]) + self.image_setting['padding_size']      # fit to the size of the segmentation bounding box
            l = min([self.image_setting['max_slice_size'], l])        # but no larger than 400 x 400

            # draw all slides,
            crop = {
                'center': [cent[1], cent[0]], # y, x because of make_grid convention
                'size': [l, l]
            }

            if bbox[5] > 12: # If there are more than 12 slid, set nrow to 4 for better view
                nrow = self.image_setting['nrow'] + 1
            else:
                nrow = self.image_setting['nrow']
            img, seg = (torch.as_tensor(sitk.GetArrayFromImage(x).astype('float')) for x in [img, seg])
            img, seg = img[display_slide_d:display_slide_u], seg[display_slide_d:display_slide_u]
            self.image = draw_grid_contour(img, [seg], color=[(255, 100, 55)],
                                           nrow=nrow, padding=2, thickness=1, crop=crop, alpha=0.8)

            return self.image

    @staticmethod
    def get_overall_diagnosis(volume,   # Unit is cm^3
                              dl_score, # not used now
                              rad_score,
                              dl_thres_func: Optional[Callable] = lambda x: x >= 0.5,
                              rad_thres_func: Optional[Callable] = lambda x: None) -> int:
        r"""
        Key:
        - [0]   Normal
        - [1]   NPC
        - [2]   Benign hyperplasia
        - [-1]   Doubtful
        """
        print(dl_score, dl_thres_func(dl_score), volume)
        if dl_thres_func(dl_score) and float(volume) >= 0.5:
            return 1 # NPC
        elif not dl_thres_func(dl_score) and float(volume) >= 3:
            return 2 # benign hyperplasia
        elif not dl_thres_func(dl_score) and float(volume) < 0.5:
            return 0 # normal
        else:
            return -1 # doubtful


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