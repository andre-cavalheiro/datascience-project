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
            ('line', plotDemStats, [('yAxes', 'standard'), ('logFile', 'shared'), ('yLabels', 'shared'),
                                    ('ymin', 'shared'), ('ymax', 'shared')]),
            ('scatter', scaterThemPlots, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                          ('annotations', 'shared')])
        ]
    },
    {
        'name': 'onlyPlotTypes',
        'type': str,
        'default': None,
        'required': False,
        'help': '',
        'possibilities': [
            ('line', plotDemStats, [('yAxes', 'standard'), ('pallets', 'shared'), ('yLabels', 'shared'),
                                    ('logFile', 'shared'), ('ymin', 'shared'), ('ymax', 'shared'),
                                    ('joinYToLabel', 'standard')]),
            ('scatter', scaterThemPlots, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'), ('yLabels', 'shared')]),
            ('scatterNx', scaterThemPlotsNx, [('ymin', 'shared'), ('ymax', 'shared'), ('logFile', 'shared'),
                                              ('yLabels', 'shared'), ('distanceToLabel', 'shared')])
        ]
    },
]