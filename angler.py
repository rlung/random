#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
import pdb


def R(angle):
    '''Rotation matrix'''
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    return rot


def angler(angle, x, y, degrees=False, plot=False):
    '''Calculate new stereotactic coordinates for angled axis'''

    # Convert to radians if necessary
    if degrees:
        angle = np.radians(angle)

    # New coordinates
    x_angled, y_angled = R(angle).dot([[x], [y]])
    x_angled, y_angled = float(x_angled), float(y_angled)
    # y_angled = np.cos(angle) * (y - x * np.tan(angle))
    # x_angled = x / np.cos(angle) + np.tan(angle) * y_angled

    if plot:
        fig, ax = plt.subplots();

        # Target
        ax.scatter([x], [y], marker='x', s=125, lw=5, c='k', label='target')
        ax.annotate(
            'old: ({:.3f}, {:.3f})\n'
            'new: ({:.3f}, {:.3f})'.format(x, y, x_angled, y_angled),
            xy=(x, y), xytext=(x * 0.8, y * 1.1), ha='right',
            arrowprops=dict(facecolor='black', width=2, headwidth=10, shrink=0.2),
        )

        # Non-angled path
        opts_old = {'ls': '--', 'lw': 2, 'c': 'k'}
        ax.plot([0, x, x], [0, 0, y], label='Normal path', **opts_old)

        # Angled path
        opts_angled = {'ls': '--', 'lw': 2}
        i, j = R(-angle).dot([[x_angled], [0]])
        ax.plot([0, i, x], [0, j, y], label='Angled path', **opts_angled)
        # ax.plot(
        #     [0, x_angled * np.cos(angle), x],
        #     [0, x_angled * -np.sin(angle), -y],
        #     label='Angled path', **opts_angled
        # )

        # Finalize figure and show
        ax.axis('equal')
        ax.legend()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()

    return x_angled, y_angled


def main():
    parser = ArgumentParser(
        description="Convert depth camera data from HDF5 to binary",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('angle', help='Angle of entry')
    parser.add_argument('x', help='x coordinate of target')
    parser.add_argument('y', help='y coordinate of target')
    parser.add_argument('-d', '--degrees', default=False, action='store_true')
    opts = parser.parse_args()

    angle = float(opts.angle)
    x = float(opts.x)
    y = float(opts.y)
    x_angled, y_angled = angler(angle, x, y, opts.degrees, plot=True)
    print('New coordinates angled at {}:\ny: {}\nx: {}'
        .format(angle, y_angled, x_angled))


if __name__ == '__main__':
    main()
