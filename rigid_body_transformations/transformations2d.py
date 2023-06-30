#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def plot_point(ax, p, frame=None, text_color='black', text_offset=0.1):
    """
    Plot a 2D point on a Matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    p : numpy.ndarray
        A 2D numpy array representing the point to plot.
    frame : str, optional
        A string representing the frame of reference for the point.
    text_color : str, optional
        The color to use for the frame label text.
    text_offset : float, optional
        The distance to offset the frame label text from the point.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> p = np.array([1, 2])
    >>> plot_point(ax, p, frame='A', text_color='red', text_offset=0.2)
    """

    # plot the point
    ax.scatter(p[0], p[1], color="black")

    if frame is not None:
        if text_color is not None:
            color = text_color
        ax.annotate(
            text=r"$" + frame + r"$",
            xy=(p[0], p[1]),
            xytext=(p[0] + text_offset, p[1] + text_offset),
            verticalalignment="top",
            horizontalalignment="left",
            color=color,
        )

class Transform2D:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        """
        Returns a string representation of the transformation.

        Returns
        -------
        str
            A string representation of the transformation.
        """
        return f"Transform2D(x={self.x}, y={self.y}, theta={self.theta})"

    def __matmul__(self, other):
        """
        Defines the behavior of the @ operator for composing transformations.

        Parameters
        ----------
        other : Transform2D
            The other transformation to compose with.

        Returns
        -------
        Transform2D
            The composed transformation.
        """
        # Define the behavior of the @ operator
        result = self.as_matrix() @ other
        return result

    def as_matrix(self):
        """
        Returns the homogeneous transformation matrix.

        Returns
        -------
        numpy.ndarray
            The 3x3 homogeneous transformation matrix.
        """
        # active rotation
        return np.array([[np.cos(self.theta), -np.sin(self.theta), self.x],
                         [np.sin(self.theta),  np.cos(self.theta), self.y],
                         [                 0,                   0,      1]])
    
    def inverse(self):
        """
        Returns the inverse transformation.

        Returns
        -------
        Transform2D
            The inverse transformation.
        """
        xy = -self.as_matrix().T[0:2, 0:2] @ self.as_matrix()[0:2, 2]
        return Transform2D(xy[0], xy[1], -self.theta)
    
    def compose(self, other):
        """
        Composes this transformation with another transformation.

        Parameters
        ----------
        other : Transform2D
            The other transformation to compose with.

        Returns
        -------
        Transform2D
            The composed transformation.
        """
        xy = self.as_matrix()[0:2, 0:2] @ np.array([other.x, other.y]) + self.as_matrix()[0:2, 2]
        return Transform2D(xy[0], xy[1], self.theta + other.theta)

    def plot(self, ax, frame=None, axis_label=True, axis_subscript=True, text_color='black', labels=('X','Y'), length=1, d1=0.05, d2=1.15):
        """
        Plot the transformation as an arrow on a Matplotlib Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to plot on.
        frame : str, optional
            A string representing the frame of reference for the transformation.
        axis_label : bool, optional
            Whether to label the x and y axes.
        axis_subscript : bool, optional
            Whether to use subscripts for the axis labels.
        text_color : str, optional
            The color to use for the frame label text.
        labels : tuple of str, optional
            The labels to use for the x and y axes.
        length : float, optional
            The length of the arrow in data units.
        d1 : float, optional
            The distance to offset the frame label text from the origin.
        d2 : float, optional
            The distance to offset the axis labels from the origin.

        Returns
        -------
        None

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> t = Transform2D(1, 2, np.pi/2)
        >>> t.plot(ax, frame='A', axis_label=True, axis_subscript=True, text_color='red', labels=('X', 'Y'), length=1, d1=0.1, d2=1.2)
        """
        # create unit vectors in homogenous coordinates
        o = self @ np.array([0, 0, 1])             # origin
        x = self @ np.array([length, 0, 1])
        y = self @ np.array([0, length, 1])

        # plot the axis
        ax.plot([o[0], x[0]], [o[1], x[1]], color="red")
        ax.plot([o[0], y[0]], [o[1], y[1]], color="lime")

        if frame is not None:
            if text_color is not None:
                color = text_color
            o1 = self @ np.array([-d1, -d1, 1])
            ax.annotate(
                text=r"$\{" + frame + r"\}$",
                xy=(o1[0], o1[1]),
                verticalalignment="top",
                horizontalalignment="left",
            )

        if axis_label:
            if text_color is not None:
                color = text_color
            # add the labels to each axis
            x = (x - o) * d2 + o
            y = (y - o) * d2 + o

            if frame is None or not axis_subscript:
                format = "${:s}$"
            else:
                format = "${:s}_{{{:s}}}$"

            ax.annotate(
                text=format.format(labels[0], frame),
                xy=(x[0], x[1]),
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.annotate(
                text=format.format(labels[1], frame),
                xy=(y[0], y[1]),
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
            )

def main():
    # Simple example of using the Transform2D class

    # Create a figure and ax[0]es
    fig, ax = plt.subplots(1, 2)

    # Reference frame A
    a = Transform2D(0, 0, np.deg2rad(0))

    # Transformation/Pose of frame B with respect
    # to frame A
    theta  = 30 # degrees
    T_ab = Transform2D(2, 1, np.deg2rad(theta))

    print('Transformation matrix of B relative to A:')
    print(T_ab.as_matrix())

    # Create a point defined in frame A
    p_a = np.array([2,2,1]) # homogeneous coordinates

    # Transform the point from frame A to frame B
    # p_b = T_ba @ p_a. Since we only have T_ab we need to invert it first.
    # This inversion is equivalent to a passive transform, which is used for
    # changing frames while keeping the point fixed.
    T_ba = T_ab.inverse()
    p_b = T_ba.as_matrix() @ p_a

    print('Point in A:')
    print(p_a[0:2])
    print('Point in B:')
    print(p_b[0:2])

    # Plot coordinate frames in  and point in frame A
    a.plot(ax=ax[0], frame="A")
    T_ab.plot(ax=ax[0], frame="B")
    plot_point(ax=ax[0], p=p_a, frame="P2")

    # Plot coordinate frame B and point in frame B
    a.plot(ax=ax[1], frame="B")
    plot_point(ax=ax[1], p=p_b, frame="P2")

    # Add labels and title
    ax[0].set_title('Frame A')
    ax[1].set_title('Frame B')

    for axis in ax:
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_aspect('equal')
        axis.grid(True)
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(-0.5, 2.5)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()