from libs.standardPlots import *

argListPlots = [
    {
        'name': 'singlePlotTypes',
        'type': str,
        'default': None,
        'required': False,
        'help': '',
        'possibilities': [
            ('line', plotDemStats, [('yAxes', 'standard'), ('logFile', 'shared'),
                                   ('ymin', 'shared'), ('ymax', 'shared')])
            ]
    },
    {
        'name': 'seqPlotTypes',
        'type': str,
        'default': None,
        'required': False,
        'help': '',
        'possibilities': [
            ('line', plotDemStats, [('yAxes', 'standard'),  ('pallets', 'shared'), ('logFile', 'shared'), ('yLabelsLine', 'standard'),
                                    ('ymin', 'shared'), ('ymax', 'shared'), ('joinYToLabel', 'standard')]),
            ('scatter', scaterThemPlots, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                          ('yLabelsScatter', 'shared'), ('annotations', 'shared')]),
            ('scatterNx', scaterThemPlotsNx, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                              ('yLabelsScatter', 'shared'), ('distanceToLabel', 'shared')]),
            ('box', plotThemBoxes, [('yAxesBox', 'standard'), ('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                              ('yLabelsBox', 'shared')])
        ]
    },
    {
        'name': 'onlyPlotTypes',
        'type': str,
        'default': None,
        'required': False,
        'help': '',
        'possibilities': [
            ('line', plotDemStats, [('yAxes', 'standard'), ('pallets', 'shared'), ('yLabelsLine', 'shared'),
                                    ('logFile', 'shared'), ('ymin', 'shared'), ('ymax', 'shared'),
                                    ('joinYToLabel', 'standard')]),
            ('scatter', scaterThemPlots, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                          ('yLabelsScatter', 'shared'), ('annotations', 'shared')]),
            ('scatterNx', scaterThemPlotsNx, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                              ('yLabelsScatter', 'shared'), ('distanceToLabel', 'shared')])
        ]
    },
]