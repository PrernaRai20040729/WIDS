import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

T = 60
P = 7
M = 4
R = 0.8
F = 0.0001
def GetData(NameOfFile):
  #START CODE HERE
  df = pd.read_csv(NameOfFile, usecols = ['ind','datadate','tic','adjcp','rsi'])




  return df 
def PartitionData():
    
    
    df=GetData('DATA.csv')
    DateToIndex = {}
    list = []

    a = df['datadate'].unique()
    i=0
    for item in a:
    
      filter = Data['datadate'] == item
      df = Data[filter]
      list.append(df)
      DateToIndex[item] = i
      i = i+1
    return list, DateToIndex 
def Switch(firstStock, SecondStock, today ,PartitionedDataFrames3):
    
    xx=pd.DataFrame(PartitionedDataFrames3[today])
    
    nf=xx.loc[(xx['tic']==firstStock) | (xx['tic']==SecondStock)]
    nf.sort_values(by=['rsi'],ascending=False)
    nf=np.array(nf)
    temp=[int(nf[0,0]%30),float(nf[0,4])]
    
    






    return temp 
def latest_prices(today):
    x,y=PartitionData()
    latest_date='20090102'
    for item in y.keys():
        if(int(item)<int(today)):
            latest_date=item
    index=y[int(latest_date)]
    a=np.array(list(x[index]['adjcp']))
    
    
    return a
def negcorr(today):
    a,b=PartitionData()
    a=np.array(a)
    columns=a[0,:,2]
    best=list(a[:today,:,3])
    best=pd.DataFrame(best)
    best.columns=[columns]
    star=best.corr()
    star=np.array(star)
    dicc={}
    for i in range(30):
        for j in range(30):
            if(star[i,j]<0):
                dicc[star[i][j]]=(i,j)
    l=list(dicc.items())
    l.sort()
    l=list(dict(l).values())
    ass=[]
    bhai=l.copy()
    for item in range(len(l)):
        for subitem in l[item]:
            if (subitem in ass):
                bhai.remove(l[item])
                break
            ass.append(subitem)
    l=bhai.copy()                
    minn=min(M,len(l))
    final_list=l[:minn]

    return final_list
class PortFolio:
    
    def __init__(self,initial_balance):
        self.ib = initial_balance
        self.cb=self.ib
        self.arr_stocks=np.zeros(30)
        x,y=PartitionData()        
        self.lp=latest_prices(list(y.keys())[T])
        self.stock_pairs=None
    #Initialize all variables

    def SellStock(self, index):
    #index : The index of the Stock to sell (0-29)
        self.cb=self.cb+((1-F)*float((self.arr_stocks[index])*self.lp[index]))
        self.arr_stocks[index]=0
  
    def BuyStock(self,index, number):
    #index : The index of the Stock to buy (0-29) 
    #number : Number of shares to buy (float)
        self.cb=self.cb-((1+F)*(number*float(self.lp[index])))
        self.arr_stocks[index]+=number

    def CalculateNetWorth(self):
    #Return Net Worth (All Shares' costs+ Balance)
        net_worth=self.cb + (sum([float(a)*float(b) for a,b in zip(self.arr_stocks,self.lp)]))
        return net_worth



    def ChangePricesTo(self,newPriceVector):
    # newPriceVector : Numpy array containing the prices of all the stocks for the current day
        self.lp=np.copy(newPriceVector)

    def ChangePairs(self,PartitionedDataFrames,today,nl):  
    # Calls the Switch function for all the pairs of stocks owned
        for index in range(30):
            myPortfolio.SellStock(index)
        indexes=[]
        for index in range(len(nl)):
            x,y=nl[index]
            u=np.array(PartitionedData[0])
            indexes.append(Switch(u[x][2],u[y][2],today,PartitionedDataFrames))
        summ=0
        for index in range(len(indexes)):
            summ+=indexes[index][1]
        for index in range(len(indexes)):
            money=indexes[index][1]*self.cb/summ
            myPortfolio.BuyStock(indexes[index][0],money/self.lp[indexes[index][0]])
        
        
            
        
        


  
    def RebalancePortfolio(self,today):
        
    
        self.stock_pairs=negcorr(today)
myPortfolio = PortFolio(int(input("Your Initial Balance?")))
NetWorthAfterEachTrade = []



Data = GetData('DATA.csv')
PartitionedData, DateToIndex= PartitionData()

#


#Start processing from the (T+1)th Day(among the ones recorded in the Data)
for i in range(T,len(list(DateToIndex.keys()))):
    # Change the Prices to the ith Term
    a=np.array(PartitionedData[i]['adjcp'])
    myPortfolio.ChangePricesTo(a)
  # Get NetWorth and store in list
    NetWorthAfterEachTrade.append(myPortfolio.CalculateNetWorth())
  # Check if you need to rebalance Portfolio's Today
    if i%T==0:
        myPortfolio.RebalancePortfolio(i)
        
  # Check if you need to switch stocks today
    if i%P==0:
        myPortfolio.ChangePairs(PartitionedData,i,myPortfolio.stock_pairs)
def VizualizeData():
    
    xpoints=np.array(range(len(NetWorthAfterEachTrade[:])))
    plt.figure(figsize=(10,5))
    plt.plot(xpoints,NetWorthAfterEachTrade[:],color='coral')
   
    plt.xlabel('Dates')
    plt.ylabel('Net Worth')
    plt.title("Net Worth Vs Date")
    plt.show()
VizualizeData()