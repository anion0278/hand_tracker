class DataLogger:
    """class for logging the data"""
    def __init__(self,filename):
        self.__filename = filename
        self.__data = []

    def log_data(self,data):
        self.__data.append(data)

    def save_data(self):
        f = open(self.__filename,"w+")
        for i in range(len(self.__data)):
            f.write(self.__data[i]+"\n")
        f.close()



