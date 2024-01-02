#Importing Required Libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
#import seaborn as sns

#Declaring the Hyperparameters

N = 50
T = 7
R = 0.8
M = 5
F = 0.0005   # 0.5% Brokerage fee
def GetData(NameOfFile):
    fields = ['datadate','tic','adjcp']
    df = pd.read_csv(NameOfFile, usecols=fields)
    return df
def PartitionData(Data):

    DateToIndex = {}
    PartitionedDataFrameList = []
    df = GetData("DATA.csv")
    grouped = df.groupby(df.datadate)

    for i in range(0,df.shape[0],30):
        PartitionedDataFrameList.append( grouped.get_group(df['datadate'][i]) )
        DateToIndex[int(df['datadate'][i])] = int(i/30)

    return PartitionedDataFrameList, DateToIndex
def GetMomentumBasedPriority(PartitionedDataFrameList, DateToIndex ,today):
  # PartitionedDataFrameList : Pandas DataFrame, The Output of your last function
  # DateToIndex : Dictionary mapping dates to their index in the PartitionedDataFrameList
  # today :  Today's date (string) In Format: YYYYMMDD


  #NdaysAgo is a datatime.date() object contining the required data, you need to convert it to a string and then check if its
  #actually there in the Data you have or will you have to get going using some other nearest date

    NdaysAgo = datetime.date(int(today[0:4]),int(today[4:6]),int(today[6:])) + datetime.timedelta(days = -N)

    df = GetData("DATA.csv")

    NdaysAgoConv = NdaysAgo.strftime('%Y%m%d')
    NdaysAgoConv = int(NdaysAgoConv)
    if NdaysAgoConv in df.datadate.values:
        n=NdaysAgoConv
    else:
        i=1
        while(True):
            NdaysAgoExp = NdaysAgo + datetime.timedelta(days = -i)
            NdaysAgoConv = NdaysAgoExp.strftime('%Y%m%d')
            NdaysAgoConv = int(NdaysAgoConv)
            if NdaysAgoConv in df.datadate.values:
                break
            NdaysAgoExp = NdaysAgo + datetime.timedelta(days = +i)
            NdaysAgoConv = NdaysAgoExp.strftime('%Y%m%d')
            NdaysAgoConv = int(NdaysAgoConv)
            if NdaysAgoConv in df.datadate.values:
                break
            i+=1
        n=NdaysAgoConv
        
    
    date_list = list(DateToIndex.keys())
    pdfl = PartitionedDataFrameList
    datesToAvgOver = [date for date in date_list if n < date <int(today)]
    numberOfDates = len(datesToAvgOver)
    sumOfPrices = np.zeros((30,))
    for date in datesToAvgOver:
        sumOfPrices = sumOfPrices + np.array(pdfl[DateToIndex[date]]['adjcp'])
    mean = sumOfPrices/numberOfDates
    priorities = (np.array(pdfl[DateToIndex[int(today)]]['adjcp'])- np.array(pdfl[DateToIndex[n]]['adjcp']))/mean
    return priorities

def GetBalanced(prices, weights,balance):
  # prices : Numpy array containing Prices of all the 30 Stocks
  # weights : Multi-hot Numpy Array : The Elements corresponding to stocks which are to be bought(Top M Stocks with positive Momentum Indicator) are set to their priority, All other elements are set to zero.
  # Returns Numpy array containing the number of shares to buy for each stock!

    sum = 0
    for i in weights:
        sum += i
    p= weights/sum
    expenditure = balance*p
    num = expenditure/prices

    return num
class PortFolio:
    def __init__(self, iniBalance, currBalance, stockNum, stockPrices):
        self.iniBalance = iniBalance
        self.currBalance = currBalance
        self.stockNum = stockNum
        self.stockPrices = stockPrices


    def SellStock(self,index):
        self.currBalance += (self.stockPrices[index]*self.stockNum[index])*(1-F)
        self.stockNum[index] = 0
    #index : The index of the Stock to sell (0-29)


    def BuyStock(self,index, number):
        self.currBalance -= self.stockPrices[index]*number*(1+F)
        self.stockNum[index] += number
    #index : The index of the Stock to buy (0-29)
    #number : Number of shares to buy (float)

    def CalculateNetWorth(self):
        stockValue = np.sum(self.stockNum*self.stockPrices)
        netWorth = stockValue + self.currBalance
        return netWorth
    #Return Net Worth (All Shares' costs+ Balance)


    def ChangePricesTo(self, newPriceVector):
        self.stockPrices = newPriceVector
    # newPriceVector : Numpy array containing the prices of all the stocks for the current day


    def RebalancePortFolio(self,newWeights):
        for i in range(0,30):
            self.SellStock(i)

        weightsList = list(newWeights)
        weightsList.sort()
        positiveNums = 0
        for weight in weightsList:
            if(weight>0):
                positiveNums+=1
        numToBuy = min(positiveNums,M)
        weightsToRetain = weightsList[30-numToBuy:30]

        weights = list(newWeights)
        for weight in weights:
            if weight in weightsToRetain:
                continue
            else:
                weight = 0
        weights = np.array(weights)

        numbers = GetBalanced(self.stockPrices, weights, self.currBalance*R)

        for i in range(0,30):
            self.BuyStock(i,numbers[i])
Data = GetData('DATA.csv')
PartitionedData, DateToIndex= PartitionData(Data)


zeroArray=np.zeros((30,))
myPortfolio = PortFolio(200,200, zeroArray , np.array(PartitionedData[N+1]['adjcp']))
NetWorthAfterEachTrade = []


#Start processing from the (N+1)th Day(among the ones recorded in the Data)
for i in range(N+1, int(len(Data['datadate'])/30)):
    # Change the Prices to the ith Term
    myPortfolio.ChangePricesTo(np.array(PartitionedData[i]['adjcp']))
    # Get NetWorth and store in list
    NetWorthAfterEachTrade.append(myPortfolio.CalculateNetWorth())
    # Check if you need to rebalance Portfolio's Today
    if((i-(N+1))%T == 0):
        newWeights = GetMomentumBasedPriority(PartitionedData, DateToIndex, str(list(DateToIndex.keys())[i]))
        myPortfolio.RebalancePortFolio(newWeights)

def VizualizeData(NetWorthList):
    n = len(NetWorthList)
    x=[]
    for i in range(0,n):
        x.append(i)
    plt.plot(x,NetWorthList)
    plt.xlabel("Days since trading start")
    plt.ylabel("Net Worth")
    plt.show()
VizualizeData(NetWorthAfterEachTrade)