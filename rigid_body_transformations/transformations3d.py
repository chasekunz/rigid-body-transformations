#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def plot_point_3d(ax, p, frame=None, text_color='black', text_offset=0.1):
    """
    Plot a 3d point on a Matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    p : numpy.ndarray
        A 3D numpy array representing the point to plot.
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
    >>> p = np.array([1, 2, 3])
    >>> plot_point(ax, p, frame='A', text_color='red', text_offset=0.2)
    """

    # Plot the point.
    ax.scatter(p[0], p[1], p[2], color="black")

    # Plot the frame label, if necessary.
    if frame is not None:
        # Use the specified color, or black by default.
        if text_color is not None:
            color = text_color
        else:
            color = 'black'

        # Add the annotation
        ax.text(
            x=p[0] + text_offset,
            y=p[1] + text_offset,
            z=p[2] + text_offset,
            s=r"$" + frame + r"$",
            verticalalignment="top",
            horizontalalignment="left",
            color=color,
        )

class Transform3D:
    def __init__(self, x, y, z, R):
        # x, y, z are the translation, R is the rotation
        self.x = x
        self.y = y
        self.z = z
        self.R = R

    def __str__(self):
        """
        Returns a string representation of the transformation.

        Returns
        -------
        str
            A string representation of the transformation.
        """
        return f"Transform3D(x={self.x}, y={self.y}, z={self.z}, R={self.R})"

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
        # Compute the matrix product
        result = self.as_matrix() @ other

        return result

    def as_matrix(self):
        """
        Returns the homogeneous transformation matrix.

        Returns
        -------
        numpy.ndarray
            The 4x4 homogeneous transformation matrix.
        """
        # Create the 4x4 matrix.
        matrix = np.eye(4)

        # Insert the rotation matrix in the upper left.
        matrix[0:3, 0:3] = self.R

        # Insert the translation in the right column.
        matrix[0:3, 3] = [self.x, self.y, self.z]

        return matrix
    
    def inverse(self):
        """
        Returns the inverse transformation.

        Returns
        -------
        Transform3d
            The inverse transformation.
        """
        # Create a new transformation.
        result = Transform3D(0, 0, 0, np.eye(3))

        # Compute the inverse rotation.
        result.R = self.R.T

        # Compute the inverse translation.
        t = -self.R.T @ np.array([self.x, self.y, self.z])
        result.x = t[0]
        result.y = t[1]
        result.z = t[2]

        return result
    
    def compose(self, other):
        """
        Composes this transformation with another transformation.

        Parameters
        ----------
        other : Transform3D
            The other transformation to compose with.

        Returns
        -------
        Transform3D
            The composed transformation.
        """
        # Compute the matrix product.
        result = self.as_matrix() @ other.as_matrix()
        return Transform3D(result[0, 3], result[1, 3], result[2, 3], result[0:3, 0:3])

    def plot(self, ax, frame=None, axis_label=True, axis_subscript=True, text_color='black', labels=('X','Y', 'Z'), length=1, d1=0.05, d2=1.15):
        """
        Plot the transformation as an arrow on a Matplotlib Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to plot on.
        frame : str, optional
            A string representing the frame of reference for the transformation.
        axis_label : bool, optional
            Whether to label the x , y, and z axes.
        axis_subscript : bool, optional
            Whether to use subscripts for the axis labels.
        text_color : str, optional
            The color to use for the frame label text.
        labels : tuple of str, optional
            The labels to use for the x, y, and z axes.
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
        x = self @ np.array([1, 0, 0, 1])
        y = self @ np.array([0, 1, 0, 1])
        z = self @ np.array([0, 0, 1, 1])
        origin = self @ np.array([0, 0, 0, 1])

        # plot the axis
        ax.plot([origin[0], x[0]], [origin[1], x[1]], [origin[2], x[2]], color="red")
        ax.plot([origin[0], y[0]], [origin[1], y[1]], [origin[2], y[2]], color="lime")
        ax.plot([origin[0], z[0]], [origin[1], z[1]], [origin[2], z[2]], color="blue")

        if frame is not None:
            if text_color is not None:
                color = text_color
            # Get the origin of the frame
            o1 = self @ np.array([-d1, -d1, -d1, 1])
            # Annotate the origin of the frame
            ax.text(
                x = o1[0],
                y = o1[1],
                z = o1[2],
                s=r"$\{" + frame + r"\}$",
                verticalalignment="top",
                horizontalalignment="left",
                color=color
            )

        if axis_label:
            if text_color is not None:
                color = text_color
            # add the labels to each axis
            x = (x - origin) * d2 + origin
            y = (y - origin) * d2 + origin
            z = (z - origin) * d2 + origin

            if frame is None or not axis_subscript:
                format = "${:s}$"
            else:
                format = "${:s}_{{{:s}}}$"

            for axis, label in zip([x, y, z], labels):
                ax.text(
                    x = axis[0],
                    y = axis[1],
                    z = axis[2],
                    s=format.format(label, frame),
                    verticalalignment="top",
                    horizontalalignment="left",
                    color=color
                )

def main():
    # Simple example of using the Transform3D class

    # Create an identity transformation of A frame
    a = Transform3D(x=0.0, y=0.0, z=0.0, R=np.eye(3))

    # Create a transformed B frame
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    b = Transform3D(x=1.0, y=-0.5, z=0.0, R=R)

    # Create a 3d point in the a frame
    p_a = np.array([2.0, 0.5, 0.25])
    print("Point in A frame:")
    print(p_a)

    # Transform the point to the b frame
    p_b = b.inverse().as_matrix() @ np.append(p_a, 1)
    print("Point in B frame:")
    print(p_b)

    # Plot the point and the two transformations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_point_3d(ax, p_a, "P")
    a.plot(ax, frame="A")
    b.plot(ax, frame="B")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis("equal")
    ax.scatter(p_a[0], p_a[1], p_a[2])
    plt.show()





if __name__ == "__main__":
    main()