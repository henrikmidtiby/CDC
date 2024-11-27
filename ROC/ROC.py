import matplotlib.pyplot as plt
import mplcursors  # type: ignore[import-not-found]
import numpy as np

from typing import Any

from numpy.typing import NDArray


class ROC:
    def __init__(self) -> None:
        self.true_positive: list[int] = []
        self.false_positive: list[int] = []
        self.true_negative: list[int] = []
        self.false_negative: list[int] = []
        self.thresholds: list[Any] = []
        self.true_positives_rate = []
        self.false_positives_rate = []
        self.precision = None

    # arguments must be numpy arrays
    def get_points(self, analysed_image, test_positive, test_negative, number_of_samples, model=None):
        if model is None:

            def model(value):
                return value

        analysed_image_temp = np.reshape(analysed_image, -1)
        test_positive_temp = np.reshape(test_positive, -1)
        test_negative_temp = np.reshape(test_negative, -1)
        thresholds = np.linspace(0, 255, number_of_samples)

        for threshold in thresholds:
            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0

            for i, value in np.ndenumerate(analysed_image_temp):
                if model(value) <= threshold:
                    if test_positive_temp[i] == 255:
                        true_positive += 1
                    if test_negative_temp[i] == 255:
                        false_positive += 1
                else:
                    if test_negative_temp[i] == 255:
                        true_negative += 1
                    if test_positive_temp[i] == 255:
                        false_negative += 1

            self.true_positive.append(true_positive)
            self.false_positive.append(false_positive)
            self.true_negative.append(true_negative)
            self.false_negative.append(false_negative)
            self.thresholds.append(threshold)

    def distance_to_points(self, analysed_image, test_positive, test_negative):
        positive_mask = np.reshape(test_positive, (-1))
        negative_mask = np.reshape(test_negative, (-1))

        distances = np.reshape(analysed_image, (-1))

        positive_distances = np.sort(distances[positive_mask >= 200])
        negative_distances = np.sort(distances[negative_mask >= 200])

        i = 0
        n = 0
        while i < len(positive_distances):
            threshold = positive_distances[i]

            if threshold < negative_distances[n]:
                true_pos = i + 1
                false_negative = len(positive_distances) - true_pos
                false_positive = n
                true_negative = len(negative_distances) - false_positive

                self.thresholds.append(threshold)
                self.true_positive.append(true_pos)
                self.false_negative.append(false_negative)
                self.false_positive.append(false_positive)
                self.true_negative.append(true_negative)
                i += 1
            else:
                n += 1

        if n < len(negative_distances):
            thresholds = negative_distances[n::10]
            for th in thresholds:
                while th > negative_distances[n]:
                    if n + 1 == len(negative_distances):
                        break
                    n += 1
                self.thresholds.append(th)

                true_pos = i + 1
                false_negative = len(positive_distances) - true_pos
                false_positive = n
                true_negative = len(negative_distances) - false_positive
                self.true_positive.append(true_pos)
                self.false_negative.append(false_negative)
                self.false_positive.append(false_positive)
                self.true_negative.append(true_negative)

        # return self.points

    def calculate_rates(self):
        self.true_positives_rate = np.divide(self.true_positive, (np.add(self.true_positive, self.false_negative)))
        self.false_positives_rate = np.divide(self.false_positive, (np.add(self.false_positive, self.true_negative)))
        self.precision = np.divide(self.true_positive, (np.add(self.true_positive, self.false_positive)))
        return 1

    def calculate_area_under_graph(self):
        area = 0
        for i in range(len(self.thresholds) - 1):
            area += (
                (self.true_positives_rate[i] + self.true_positives_rate[i + 1])
                / 2
                * (self.false_positives_rate[i + 1] - self.false_positives_rate[i])
            )
        return area

    def plot_ROC(self, options):
        match options:
            case None | "FPR":
                x = self.false_positives_rate
                x_label = "False Positive Rate"
                y = self.true_positives_rate
                y_label = "True Positive Rate"
            case "precision":
                x = self.true_positives_rate
                x_label = "True Positive Rate"
                y = self.precision
                y_label = "Precision"
            case _:
                print("Option used didn't match any implemented option for x axis value, False Positive Rate was used")
                x = self.false_positives_rate
                x_label = "False Positive Rate"

        y = self.true_positives_rate
        z = self.thresholds
        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=z, cmap="viridis")
        plt.fill_between(self.false_positives_rate, self.true_positives_rate, color="red", alpha=0.4)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("ROC Curve")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        cursor = mplcursors.cursor(sc, hover=True)

        # Define what happens when hovering over a point
        @cursor.connect("add")
        def on_add(sel):
            # Display the value from the threshold array
            sel.annotation.set(text=f"threshold={self.thresholds[sel.index]:.3f}")

        #
        plt.show()
