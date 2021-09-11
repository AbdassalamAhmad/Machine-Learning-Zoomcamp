```python
import numpy as np
import pandas as pd
```

# Question 1


```python
np.__version__
```




    '1.20.1'



# Question 2


```python
pd.__version__

```




    '1.2.4'



Viewing our Dataset


```python
df=pd.read_csv("E:\machine learning\data.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11909</th>
      <td>Acura</td>
      <td>ZDX</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Crossover,Hatchback,Luxury</td>
      <td>Midsize</td>
      <td>4dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>46120</td>
    </tr>
    <tr>
      <th>11910</th>
      <td>Acura</td>
      <td>ZDX</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Crossover,Hatchback,Luxury</td>
      <td>Midsize</td>
      <td>4dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>56670</td>
    </tr>
    <tr>
      <th>11911</th>
      <td>Acura</td>
      <td>ZDX</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Crossover,Hatchback,Luxury</td>
      <td>Midsize</td>
      <td>4dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>50620</td>
    </tr>
    <tr>
      <th>11912</th>
      <td>Acura</td>
      <td>ZDX</td>
      <td>2013</td>
      <td>premium unleaded (recommended)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Crossover,Hatchback,Luxury</td>
      <td>Midsize</td>
      <td>4dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>204</td>
      <td>50920</td>
    </tr>
    <tr>
      <th>11913</th>
      <td>Lincoln</td>
      <td>Zephyr</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>221.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Luxury</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>26</td>
      <td>17</td>
      <td>61</td>
      <td>28995</td>
    </tr>
  </tbody>
</table>
<p>11914 rows × 16 columns</p>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe().round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11914.00</td>
      <td>11845.00</td>
      <td>11884.00</td>
      <td>11908.00</td>
      <td>11914.00</td>
      <td>11914.00</td>
      <td>11914.00</td>
      <td>11914.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2010.38</td>
      <td>249.39</td>
      <td>5.63</td>
      <td>3.44</td>
      <td>26.64</td>
      <td>19.73</td>
      <td>1554.91</td>
      <td>40594.74</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.58</td>
      <td>109.19</td>
      <td>1.78</td>
      <td>0.88</td>
      <td>8.86</td>
      <td>8.99</td>
      <td>1441.86</td>
      <td>60109.10</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.00</td>
      <td>55.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>12.00</td>
      <td>7.00</td>
      <td>2.00</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2007.00</td>
      <td>170.00</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>22.00</td>
      <td>16.00</td>
      <td>549.00</td>
      <td>21000.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.00</td>
      <td>227.00</td>
      <td>6.00</td>
      <td>4.00</td>
      <td>26.00</td>
      <td>18.00</td>
      <td>1385.00</td>
      <td>29995.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.00</td>
      <td>300.00</td>
      <td>6.00</td>
      <td>4.00</td>
      <td>30.00</td>
      <td>22.00</td>
      <td>2009.00</td>
      <td>42231.25</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2017.00</td>
      <td>1001.00</td>
      <td>16.00</td>
      <td>4.00</td>
      <td>354.00</td>
      <td>137.00</td>
      <td>5657.00</td>
      <td>2065902.00</td>
    </tr>
  </tbody>
</table>
</div>



# Question 3


```python
df.groupby("Make").MSRP.mean().round(3)
```




    Make
    Acura              34887.587
    Alfa Romeo         61600.000
    Aston Martin      197910.376
    Audi               53452.113
    BMW                61546.763
    Bentley           247169.324
    Bugatti          1757223.667
    Buick              28206.612
    Cadillac           56231.317
    Chevrolet          28350.386
    Chrysler           26722.963
    Dodge              22390.059
    FIAT               22670.242
    Ferrari           238218.841
    Ford               27399.267
    GMC                30493.299
    Genesis            46616.667
    HUMMER             36464.412
    Honda              26674.341
    Hyundai            24597.036
    Infiniti           42394.212
    Kia                25310.173
    Lamborghini       331567.308
    Land Rover         67823.217
    Lexus              47549.069
    Lincoln            42839.829
    Lotus              69188.276
    Maserati          114207.707
    Maybach           546221.875
    Mazda              20039.383
    McLaren           239805.000
    Mercedes-Benz      71476.229
    Mitsubishi         21240.535
    Nissan             28583.432
    Oldsmobile         11542.540
    Plymouth            3122.902
    Pontiac            19321.548
    Porsche           101622.397
    Rolls-Royce       351130.645
    Saab               27413.505
    Scion              19932.500
    Spyker            213323.333
    Subaru             24827.504
    Suzuki             17907.208
    Tesla              85255.556
    Toyota             29030.016
    Volkswagen         28102.381
    Volvo              28541.160
    Name: MSRP, dtype: float64




```python
BMW_mean=df.loc[df['Make'] == 'BMW', 'MSRP'].mean()
BMW_mean
```




    61546.76347305389



# Question 4


```python
cars_after_2015=df[
                    df['Year'] >=2015
                  ]
cars_after_2015
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>FIAT</td>
      <td>124 Spider</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>35</td>
      <td>26</td>
      <td>819</td>
      <td>27495</td>
    </tr>
    <tr>
      <th>33</th>
      <td>FIAT</td>
      <td>124 Spider</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>35</td>
      <td>26</td>
      <td>819</td>
      <td>24995</td>
    </tr>
    <tr>
      <th>34</th>
      <td>FIAT</td>
      <td>124 Spider</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>35</td>
      <td>26</td>
      <td>819</td>
      <td>28195</td>
    </tr>
    <tr>
      <th>41</th>
      <td>BMW</td>
      <td>2 Series</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>240.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>35</td>
      <td>23</td>
      <td>3916</td>
      <td>32850</td>
    </tr>
    <tr>
      <th>42</th>
      <td>BMW</td>
      <td>2 Series</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>240.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>34</td>
      <td>23</td>
      <td>3916</td>
      <td>38650</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11898</th>
      <td>BMW</td>
      <td>Z4</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>17</td>
      <td>3916</td>
      <td>56950</td>
    </tr>
    <tr>
      <th>11899</th>
      <td>BMW</td>
      <td>Z4</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>17</td>
      <td>3916</td>
      <td>65800</td>
    </tr>
    <tr>
      <th>11900</th>
      <td>BMW</td>
      <td>Z4</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>17</td>
      <td>3916</td>
      <td>57500</td>
    </tr>
    <tr>
      <th>11901</th>
      <td>BMW</td>
      <td>Z4</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>240.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>34</td>
      <td>22</td>
      <td>3916</td>
      <td>49700</td>
    </tr>
    <tr>
      <th>11902</th>
      <td>BMW</td>
      <td>Z4</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>17</td>
      <td>3916</td>
      <td>66350</td>
    </tr>
  </tbody>
</table>
<p>5995 rows × 16 columns</p>
</div>




```python
missings=cars_after_2015.isnull().sum()
missings
```




    Make                    0
    Model                   0
    Year                    0
    Engine Fuel Type        0
    Engine HP              51
    Engine Cylinders        8
    Transmission Type       0
    Driven_Wheels           0
    Number of Doors         5
    Market Category      1324
    Vehicle Size            0
    Vehicle Style           0
    highway MPG             0
    city mpg                0
    Popularity              0
    MSRP                    0
    dtype: int64




```python
missings['Engine HP']
```




    51



# Question 5


```python
mean_1=df["Engine HP"].mean()
mean_1
```




    249.38607007176023




```python
df["Engine HP"]=df["Engine HP"].fillna(61546.76347305389)
df["Engine HP"]
```




    0        335.0
    1        300.0
    2        300.0
    3        230.0
    4        230.0
             ...  
    11909    300.0
    11910    300.0
    11911    300.0
    11912    300.0
    11913    221.0
    Name: Engine HP, Length: 11914, dtype: float64




```python
df['Engine HP'].isnull().sum()
```




    0




```python
mean_2=df["Engine HP"].mean()
mean_2
```




    604.3901863052465




```python
print(round(mean_1))
print(round(mean_2))

```

    249
    604
    

# Question 6


```python
newdf=df
newdf=newdf.loc[newdf["Make"]=="Rolls-Royce"]
newdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2921</th>
      <td>Rolls-Royce</td>
      <td>Corniche</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>325.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury</td>
      <td>Large</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>86</td>
      <td>359990</td>
    </tr>
    <tr>
      <th>3505</th>
      <td>Rolls-Royce</td>
      <td>Dawn</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,High-Performance</td>
      <td>Large</td>
      <td>Convertible</td>
      <td>19</td>
      <td>12</td>
      <td>86</td>
      <td>335000</td>
    </tr>
    <tr>
      <th>5275</th>
      <td>Rolls-Royce</td>
      <td>Ghost Series II</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>286750</td>
    </tr>
    <tr>
      <th>5276</th>
      <td>Rolls-Royce</td>
      <td>Ghost Series II</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>319400</td>
    </tr>
    <tr>
      <th>5277</th>
      <td>Rolls-Royce</td>
      <td>Ghost Series II</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>295850</td>
    </tr>
    <tr>
      <th>5278</th>
      <td>Rolls-Royce</td>
      <td>Ghost Series II</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>329325</td>
    </tr>
    <tr>
      <th>5279</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>20</td>
      <td>13</td>
      <td>86</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>5280</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>20</td>
      <td>13</td>
      <td>86</td>
      <td>290000</td>
    </tr>
    <tr>
      <th>5281</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>296000</td>
    </tr>
    <tr>
      <th>5282</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>256650</td>
    </tr>
    <tr>
      <th>5283</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>263200</td>
    </tr>
    <tr>
      <th>5284</th>
      <td>Rolls-Royce</td>
      <td>Ghost</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>563.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>303300</td>
    </tr>
    <tr>
      <th>7443</th>
      <td>Rolls-Royce</td>
      <td>Park Ward</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>322.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>15</td>
      <td>11</td>
      <td>86</td>
      <td>259900</td>
    </tr>
    <tr>
      <th>7444</th>
      <td>Rolls-Royce</td>
      <td>Park Ward</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>322.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>15</td>
      <td>11</td>
      <td>86</td>
      <td>262990</td>
    </tr>
    <tr>
      <th>7553</th>
      <td>Rolls-Royce</td>
      <td>Phantom Coupe</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>433550</td>
    </tr>
    <tr>
      <th>7554</th>
      <td>Rolls-Royce</td>
      <td>Phantom Coupe</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>438325</td>
    </tr>
    <tr>
      <th>7555</th>
      <td>Rolls-Royce</td>
      <td>Phantom Coupe</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>449525</td>
    </tr>
    <tr>
      <th>7556</th>
      <td>Rolls-Royce</td>
      <td>Phantom Drophead Coupe</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Convertible</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>474600</td>
    </tr>
    <tr>
      <th>7557</th>
      <td>Rolls-Royce</td>
      <td>Phantom Drophead Coupe</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Convertible</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>479775</td>
    </tr>
    <tr>
      <th>7558</th>
      <td>Rolls-Royce</td>
      <td>Phantom Drophead Coupe</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Convertible</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>492000</td>
    </tr>
    <tr>
      <th>7559</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>402940</td>
    </tr>
    <tr>
      <th>7560</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>474990</td>
    </tr>
    <tr>
      <th>7561</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>407400</td>
    </tr>
    <tr>
      <th>7562</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>480175</td>
    </tr>
    <tr>
      <th>7563</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>417825</td>
    </tr>
    <tr>
      <th>7564</th>
      <td>Rolls-Royce</td>
      <td>Phantom</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>453.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury,Performance</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>19</td>
      <td>11</td>
      <td>86</td>
      <td>492425</td>
    </tr>
    <tr>
      <th>9431</th>
      <td>Rolls-Royce</td>
      <td>Silver Seraph</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>322.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>15</td>
      <td>11</td>
      <td>86</td>
      <td>219900</td>
    </tr>
    <tr>
      <th>9432</th>
      <td>Rolls-Royce</td>
      <td>Silver Seraph</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>322.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Exotic,Luxury</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>15</td>
      <td>11</td>
      <td>86</td>
      <td>229990</td>
    </tr>
    <tr>
      <th>11448</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,High-Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>284900</td>
    </tr>
    <tr>
      <th>11449</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,High-Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>294025</td>
    </tr>
    <tr>
      <th>11450</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Exotic,Luxury,High-Performance</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>304350</td>
    </tr>
  </tbody>
</table>
</div>




```python
X1=newdf[["Engine HP", "Engine Cylinders", "highway MPG"]]
X1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>highway MPG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2921</th>
      <td>325.0</td>
      <td>8.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3505</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5275</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5276</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5277</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5278</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5279</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5280</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5281</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5282</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5283</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5284</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7443</th>
      <td>322.0</td>
      <td>12.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7444</th>
      <td>322.0</td>
      <td>12.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7553</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7554</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7555</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7556</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7557</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7558</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7559</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7560</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7561</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7562</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7563</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7564</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9431</th>
      <td>322.0</td>
      <td>12.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>9432</th>
      <td>322.0</td>
      <td>12.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>11448</th>
      <td>624.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11449</th>
      <td>624.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>11450</th>
      <td>624.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
X1=X1.drop_duplicates()
X1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>highway MPG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2921</th>
      <td>325.0</td>
      <td>8.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3505</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5275</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5279</th>
      <td>563.0</td>
      <td>12.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7443</th>
      <td>322.0</td>
      <td>12.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7553</th>
      <td>453.0</td>
      <td>12.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>11448</th>
      <td>624.0</td>
      <td>12.0</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=np.array(X1)
X
```




    array([[325.,   8.,  15.],
           [563.,  12.,  19.],
           [563.,  12.,  21.],
           [563.,  12.,  20.],
           [322.,  12.,  15.],
           [453.,  12.,  19.],
           [624.,  12.,  21.]])




```python
XT=X.T
XT
```




    array([[325., 563., 563., 563., 322., 453., 624.],
           [  8.,  12.,  12.,  12.,  12.,  12.,  12.],
           [ 15.,  19.,  21.,  20.,  15.,  19.,  21.]])




```python
XTX=XT.dot(X)
XTX
```




    array([[1.754801e+06, 3.965600e+04, 6.519600e+04],
           [3.965600e+04, 9.280000e+02, 1.500000e+03],
           [6.519600e+04, 1.500000e+03, 2.454000e+03]])




```python
XTXinv=np.linalg.inv(XTX)
XTXinv
```




    array([[ 5.17815728e-05,  9.06587044e-04, -1.92984188e-03],
           [ 9.06587044e-04,  1.05723058e-01, -8.87084092e-02],
           [-1.92984188e-03, -8.87084092e-02,  1.05900809e-01]])




```python
XTXinv.sum()
```




    0.032212320677486195



# Question 7


```python
y=np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
y
```




    array([1000, 1100,  900, 1200, 1000,  850, 1300])




```python
w=(XTXinv.dot(XT)).dot(y)
w
```




    array([ 0.19989598, 31.02612262, 31.65378877])




```python
w[0]
```




    0.19989598183186175




```python

```
