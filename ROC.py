import numpy as np
import matplotlib.pyplot as plt 
import mplcursors

class ROC:
    def __init__(self) -> None:
        self.true_positive = []
        self.false_positive = []
        self.true_negative = []
        self.false_negative = []
        self.thresholds=[]
        self.true_positives_rate=None
        self.false_positives_rate=None
        self.precission=None 



    def get_points(self,analysed_image,test_positive,test_negative,number_of_samples,model=None):
        if model==None:
            def model(value):
                return value
        
        analysed_image_temp = np.reshape(analysed_image,-1)
        test_positive_temp  = np.reshape(test_positive,-1)
        test_negative_temp  = np.reshape(test_negative,-1)
        thresholds          = np.linspace(0,255,number_of_samples)

        for threshold in thresholds:
            true_positive       = 0
            false_positive      = 0
            true_negative       = 0
            false_negative      = 0
        
            for i, value in np.ndenumerate(analysed_image_temp):
                if model(value)     <=   threshold:
                    if test_positive_temp[i]    ==  255:
                        true_positive   +=  1
                    if test_negative_temp[i]    ==  255:
                        false_positive  +=  1
                else:
                    if test_negative_temp[i]    ==  255:
                        true_negative   +=  1
                    if test_positive_temp[i]    ==  255:
                        false_negative  +=  1


            self.true_positive.append(true_positive)
            self.false_positive.append(false_positive)
            self.true_negative.append(true_negative)
            self.false_negative.append(false_negative)
            self.thresholds.append(threshold)
    
        #return self.points

    def calculate_rates(self):
        self.true_positives_rate = np.divide(self.true_positive , ( np.add(self.true_positive , self.false_negative )))
        self.false_positives_rate= np.divide(self.false_positive , ( np.add(self.false_positive , self.true_negative )))
        self.precission = np.divide( self.true_positive , ( np.add(self.true_positive , self.false_positive )))
        return 1
    
    def calculate_area_under_graph(self):
        area = 0
        for i in range(len(self.thresholds)-1):
           area += ( self.true_positives_rate[i] + self.true_positives_rate[i+1] ) / 2 * ( self.false_positives_rate[i+1] - self.false_positives_rate[i] ) 
        return area

    def plot_ROC(self,options=None):
        match options:
            case None | 'FPR':
                x=self.false_positives_rate
            case 'precision':
                x=self.precission
            case _:
                print('Option used didnt match any implemented option for x axis value, False Positive Rate was used')
                x=self.false_positives_rate
                
        y=self.true_positives_rate
        z=self.thresholds
        fig, ax= plt.subplots()
        sc= ax.scatter(x , y , c=z, cmap='viridis')
        plt.fill_between(self.false_positives_rate , self.true_positives_rate, color='red',alpha=0.4)
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("ROC Curve")
        plt.xlabel("False Positives")
        plt.ylabel("True Positives")

        cursor = mplcursors.cursor(sc, hover=True)

    # Define what happens when hovering over a point
        @cursor.connect("add")
        def on_add(sel):
    # Display the value from the threshold array
            sel.annotation.set(text=f'threshold={self.thresholds[sel.index]:.3f}')

    #
        plt.show()

    
    
    