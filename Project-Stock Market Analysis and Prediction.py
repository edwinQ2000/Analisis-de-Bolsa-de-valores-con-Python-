
# coding: utf-8

# Proyecto: Análisis y predicción bursátil
#
# Datos bursátiles de Google Finance
#
# ### Introducción
#
# Análisis y predicción bursátil es el proyecto de análisis técnico,
# visualización y predicción utilizando los datos proporcionados por Google Finance.
# Observando los datos del mercado de valores, en particular algunos gigantes de la tecnología
# acciones y otros. Utilizando pandas para obtener información de las acciones, visualizar diferentes aspectos de la misma, y finalmente buscar algunas formas de analizar el riesgo de una acción, en base a su historial de rendimiento anterior. Predijo los precios futuros de las acciones a través de un método de Monte Carlo.
#

# In[1]:
# Configuracion inicial
# Para la division
from __future__ import division
from datetime import datetime
from pandas_datareader import DataReader
from IPython import get_ipython
from IPython.display import SVG


# Para procesar los datos
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Para la visualizacion
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().magic(u'matplotlib inline')
#get_ipython().magic('matplotlib inline')

# In[2]:

# Para leer los datos del mercado desde Yahoo

# Para  el tiempo de captura de datos


# #### Section 1 - Analisis basico de la informacion de la bolsa de valores
#
# En esta sección, repasaré cómo manejar la solicitud de información de acciones con pandas y cómo analizar los atributos básicos de una acción.
#
#

# In[3]:


# Lista de Tech_stocks para el analisis
tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# inicio y finalizacion de grabado de la informacion
end = datetime.now()
start = datetime(end.year-1, end.month, end.day)


# ciclo para grabar los datos de las finanzas de google y la configuracion de los dataframe

# Asignar el tablero de cotizaciones en el dataframe

for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo', start, end)


#

# In[4]:


AAPL.head()


# In[5]:


# Estadísticas resumidas de las acciones de Apple
AAPL.describe()


# In[6]:


# Informacion general
AAPL.info()


# Ahora que hemos visto el DataFrame, vamos a trazar el volumen y el precio de cierre de las acciones de AAPL(Apple).
# In[7]:


# Veamos una visión histórica del precio de cierre
AAPL['Close'].plot(legend=True, figsize=(10, 4))


# In[8]:


# Ahora vamos a trazar el volumen total de acciones que se negocian cada día en el último año

AAPL['Volume'].plot(legend=True, figsize=(10, 4))


# Podemos ver que en Enero'2022 fue el más alto para las acciones de AAPL que se negocian.

# Ahora que hemos visto las visualizaciones para el precio de cierre y el volumen negociado cada día para las acciones de AAPL.
# Vamos a calcular la media móvil de las acciones de AAPL.
#

# Para más información sobre las medias móviles (SMA y EMA) consulte los siguientes enlaces:
#
# 1.) http://www.investopedia.com/terms/m/movingaverage.asp
#
# 2.) http://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp

# In[9]:


# Pandas tiene una calculadora de medias móviles incorporada

# Vamos a trazar varias medias móviles
MA_day = [10, 20, 50, 100]

for ma in MA_day:
    column_name = 'MA for %s days' % (str(ma))
    #AAPL[column_name] = pd.rolling_mean(AAPL['Close'],ma)
    AAPL[column_name] = AAPL['Close'].rolling(ma).mean()


# In[10]:
# Ahora, vamos a trazar todas las medias móviles adicionales para las acciones de AAPL


AAPL[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days',
      'MA for 100 days']].plot(subplots=False, figsize=(10, 4))


# #### Sección 2 - Análisis de la rentabilidad diaria
#

# Ahora que hemos hecho un análisis de referencia, vamos a profundizar un poco más. Ahora vamos a analizar el riesgo de la acción.
# Para ello, tenemos que echar un vistazo a los cambios diarios de la acción, y no sólo su valor absoluto. Vamos a utilizar pandas para recuperar los rendimientos diarios de la acción APPL.
#

# In[11]:


# Utilizaremos pct_change para encontrar el porcentaje de cambio de cada día
AAPL['Daily Return'] = AAPL['Close'].pct_change()

# Trazamos el porcentaje de retorno diario
AAPL['Daily Return'].plot(figsize=(12, 4), legend=True,
                          linestyle='--', marker='o')


# ahora vamos a echar un vistazo general a la rentabilidad media diaria utilizando un histograma. Usando seaborn para crear tanto un histograma como un gráfico kde en la misma figura.

# In[12]:


# solo con el histograma
AAPL['Daily Return'].hist(bins=100)


# In[13]:


# Tenga en cuenta el uso de dropna() aquí, de lo contrario los valores NaN no pueden ser leídos por seaborn
sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='magenta')


# ¿Y si queremos analizar los rendimientos de todas las acciones de nuestra lista? Para ello, necesitamos construir un DataFrame con todas las columnas ['Close'] para cada uno de los dataframes de las acciones.

# In[14]:


# Recoge todos los precios de cierre de la lista de acciones tecnológicas en un DataFrame

closingprice_df = DataReader(tech_list, 'yahoo', start, end)['Close']


# In[15]:


closingprice_df.head(10)


# Ahora que tenemos todos los precios de cierre, vamos a obtener la rentabilidad diaria de todas las acciones, como hicimos con la acción APPL.

# In[16]:


# hacer una nueva tech devuelve DataFrame
tech_returns = closingprice_df.pct_change()


# In[17]:


tech_returns.head()


# Ahora podemos comparar la rentabilidad porcentual diaria de dos valores para comprobar su correlación. Primero veamos una acción comparada consigo misma.
#
# ##### GOOGL es una acción de clase A de Alphabet Inc.

# In[18]:


# La comparación de Google con sí mismo debería mostrar una relación perfectamente lineal
sns.jointplot('GOOGL', 'GOOGL', tech_returns, kind='scatter', color='orange')


# Así que ahora podemos ver que si dos acciones están perfectamente (y positivamente) correlacionadas entre sí, debería producirse una relación lineal entre sus valores de rendimiento diario.
#
# Así que vamos a seguir adelante y comparar Google y Amazon de la misma manera.

# In[19]:


# Utilizaremos joinplot para comparar los rendimientos diarios de Google y Amazon.

sns.jointplot('GOOGL', 'AMZN', tech_returns,
              kind='scatter', size=8, color='skyblue')


# In[20]:


# con la trama Hex
sns.jointplot('GOOGL', 'AMZN', tech_returns,
              kind='hex', size=8, color='skyblue')


# In[21]:


# Comprobemos si Apple y Microsoft tienen un gráfico conjunto
sns.jointplot('AAPL', 'MSFT', tech_returns,
              kind='reg', size=8, color='skyblue')


# Es interesante que el valor pearsonr (conocido oficialmente como el coeficiente de correlación producto-momento de Pearson) pueda darle una idea de cómo están correlacionados los rendimientos porcentuales diarios. Puede encontrar más información al respecto en este enlace:
#
# Url - http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
#
# Pero para tener una idea rápida e intuitiva, mira la imagen de abajo.

# In[22]:


SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')


# Seaborn y Pandas hacen que sea muy fácil repetir este análisis de comparación para cada combinación posible de valores en nuestra lista de valores tecnológicos. Podemos utilizar sns.pairplot() para crear automáticamente este gráfico

# In[23]:


# Podemos simplemente llamar a pairplot en nuestro DataFrame para un análisis visual automático de todas las comparaciones
sns.pairplot(tech_returns.dropna(), size=3)


# Arriba podemos ver todas las relaciones en los rendimientos diarios entre todas las acciones. Un rápido vistazo muestra una interesante correlación entre los rendimientos diarios de Google y Amazon. Podría ser interesante investigar esa comparación individual. Aunque la simplicidad de llamar simplemente a sns.pairplot() es fantástica, también podemos utilizar sns.PairGrid() para tener un control total de la figura, incluyendo qué tipo de gráficos van en la diagonal, el triángulo superior y el triángulo inferior.
#
# A continuación se muestra un ejemplo de utilización de toda la potencia de seaborn para lograr este resultado.


# In[24]:


# Configure la figura nombrándola returns_fig, llame a PairGrid en el DataFrame
returns_fig = sns.PairGrid(tech_returns.dropna())

# Usando map_upper podemos especificar cómo será el triángulo superior.
returns_fig.map_upper(plt.scatter, color='purple')

# También podemos definir el triángulo inferior de la figura, incluyendo el tipo de trazado (kde) y el mapa de colores (AzulPúrpura)
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Por último, definiremos la diagonal como una serie de histogramas de la rentabilidad diaria
returns_fig.map_diag(plt.hist, bins=30)


# También podemos analizar la correlación de los precios de cierre utilizando exactamente esta misma técnica. Aquí se muestra, el código repetido de arriba con la excepción del DataFrame llamado.

# In[25]:


# Configure la figura nombrándola returns_fig, llame a PairGrid en el DataFrame
returns_fig = sns.PairGrid(closingprice_df.dropna())

# Usando map_upper podemos especificar cómo será el triángulo superior.
returns_fig.map_upper(plt.scatter, color='purple')

# También podemos definir el triángulo inferior de la figura, incluyendo el tipo de trazado (kde) y el mapa de colores (AzulPúrpura)
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Por último, definiremos la diagonal como una serie de histogramas de la rentabilidad diaria
returns_fig.map_diag(plt.hist, bins=30)


# Por último, también podemos hacer un gráfico de correlación, para obtener valores numéricos reales de la correlación entre los valores de rendimiento diario de las acciones. Al comparar los precios de cierre, vemos una relación interesante entre las acciones de Google y Amazon.

# In[26]:


# Sigamos adelante y utilicemos seaborn para un rápido mapa de calor para obtener la correlación para el retorno diario de las acciones.
sns.heatmap(tech_returns.corr(), annot=True, fmt=".3g", cmap='YlGnBu')


# In[27]:


# Comprobemos la correlación entre los precios de cierre de las acciones
sns.heatmap(closingprice_df.corr(), annot=True, fmt=".3g", cmap='YlGnBu')


# #####  Tal y como sospechábamos en nuestro PairPlot, aquí vemos numérica y visualmente que Amazon y Google tienen la mayor correlación de rentabilidad bursátil diaria. También es interesante ver que todas las empresas tecnológicas están correlacionadas positivamente.

# Ahora que hemos hecho un análisis de la rentabilidad diaria, vamos a empezar a profundizar en el análisis del riesgo real.

# Análisis de Riesgo

# Hay muchas maneras de cuantificar el riesgo, una de las más básicas utilizando la información que hemos recopilado sobre los rendimientos porcentuales diarios es comparando el rendimiento esperado con la desviación estándar de los rendimientos diarios (Riesgo).

# In[28]:


# Empecemos por definir un nuevo DataFrame como una versión clenaed del DataFrame oriignal tech_returns
rets = tech_returns.dropna()


# In[29]:


rets.head()


# In[30]:


# Definir el área de los círculos del gráfico de dispersión para evitar los pequeños puntos
area = np.pi*20

plt.scatter(rets.mean(), rets.std(), s=area)

# Establezca los límites x e y del gráfico (opcional, elimine esto si no ve nada en su gráfico)
plt.xlim([-0.0025, 0.0025])
plt.ylim([0.001, 0.025])

# Establecer los títulos de los ejes de trazado
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# Etiquetar los gráficos de dispersión, para más información sobre cómo se hace, consulte el siguiente enlace
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy=(x, y), xytext=(50, 50),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='fancy', connectionstyle='arc3,rad=-0.3'))


# Si observamos el gráfico de dispersión, podemos decir que estas acciones tienen un riesgo menor y una rentabilidad esperada positiva.

# Valor en riesgo

# Vamos a definir un parámetro de valor en riesgo para nuestras acciones. Podemos tratar el valor en riesgo como la cantidad de dinero que podríamos esperar perder (es decir, poner en riesgo) para un intervalo de confianza determinado. Hay varios métodos que podemos utilizar para estimar el valor en riesgo. Veamos algunos de ellos en acción.
#
# #### Valor en riesgo utilizando el método "bootstrap
# Para este método calcularemos los cuantiles empíricos a partir de un histograma de rentabilidades diarias. Para más información sobre los cuantiles, consulte este enlace: http://en.wikipedia.org/wiki/Quantile
#
# Vamos a repetir el histograma de rendimientos diarios para las acciones de Apple.

# In[31]:


# Tenga en cuenta el uso de dropna() aquí, de lo contrario los valores NaN no pueden ser leídos por seaborn
sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')


# Ahora podemos utilizar el cuantil para obtener el valor de riesgo de la acción.

# In[32]:


# El cuantil empírico 0,05 de los rendimientos diarios

# Para las acciones APPL
rets["AAPL"].quantile(0.05)


# El cuantil empírico 0,05 de los rendimientos diarios está en -0,016. Esto significa que, con un 95% de confianza, nuestra peor pérdida diaria no superará el 1,6%. Si tenemos una inversión de 1 millón de dólares, nuestro VaR de un día del 5% es de 0,016 * 1.000.000 = 16.000 dólares.

# In[33]:


# Para las acciones de AMZN
rets["AMZN"].quantile(0.05)


# In[34]:


# Para las acciones de GOOGL
rets["GOOGL"].quantile(0.05)


# In[35]:


# Para las acciones de MSFT
rets["MSFT"].quantile(0.05)


# #### Valor en riesgo utilizando el método de Montecarlo
# Usando el Monte Carlo para realizar muchos ensayos con condiciones de mercado aleatorias, entonces calcularemos las pérdidas de la cartera para cada ensayo. Después de esto, utilizaremos la agregación de todas estas simulaciones para establecer el grado de riesgo de la acción.
#
# Comencemos con una breve explicación de lo que vamos a hacer:
#
# Utilizaremos el movimiento browniano geométrico (GBM), que técnicamente se conoce como proceso de Markov. Esto significa que el precio de las acciones sigue un paseo aleatorio y es consistente con (al menos) la forma débil de la hipótesis del mercado eficiente (HME): la información del precio pasado ya está incorporada y el siguiente movimiento del precio es "condicionalmente independiente" de los movimientos del precio pasado.
#
# Esto significa que la información pasada sobre el precio de una acción es independiente de dónde estará el precio de la acción en el futuro, lo que significa básicamente que no se puede predecir perfectamente el futuro basándose únicamente en el precio anterior de una acción.
#

# Ahora vemos que el cambio en el precio de la acción es el precio actual de la acción multiplicado por dos términos. El primer término se conoce como "deriva", que es la rentabilidad media diaria multiplicada por el cambio de tiempo. El segundo término se conoce como "shock", para cada periodo de tiempo la acción "derivará" y luego experimentará un "shock" que empujará aleatoriamente el precio de la acción hacia arriba o hacia abajo. Simulando esta serie de pasos de deriva y choque miles de veces, podemos empezar a hacer una simulación de dónde podríamos esperar que estuviera el precio de las acciones.
#
# Para más información sobre el método Monte Carlo para acciones y la simulación de los precios de las acciones con el modelo GBM, es decir, el movimiento browniano geométrico (GBM).
#
# consulte el siguiente enlace: http://www.investopedia.com/articles/07/montecarlo.asp

# Para demostrar un método básico de Monte Carlo, empezaremos con unas pocas simulaciones. Primero definiremos las variables que usaremos en el DataFrame de las acciones de Google GOOGL
#

# In[36]:


rets.head()


# In[37]:


# Establecer nuestro horizonte temporal
days = 365

# Ahora nuestro delta
dt = 1/days

# Ahora tomemos nuestra mu (deriva) de los datos de retorno esperado que obtuvimos para GOOGL
mu = rets.mean()['GOOGL']

# Ahora tomemos la volatilidad de la acción a partir de la std() de la rentabilidad media de GOOGL
sigma = rets.std()['GOOGL']


# A continuación, crearemos una función que toma el precio inicial y el número de días, y utiliza el sigma y el mu que ya hemos calculado a partir de nuestros rendimientos diarios.

# In[38]:


def stock_monte_carlo(start_price, days, mu, sigma):
    ''' Esta función toma el precio inicial de las acciones, los días de simulación, mu, sigma, y devuelve la matriz de precios simulados'''

    # Definir una matriz de precios
    price = np.zeros(days)
    price[0] = start_price

    # Schok y Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Ejecutar la matriz de precios para el número de días
    for x in range(1, days):

        # Calcular Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calcular Drift
        drift[x] = mu * dt
        # Calcular Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price

    # Ahora vamos a poner la función anterior a trabajar.

# In[39]:


# Para las acciones de Google - GOOGL
GOOGL.head()


# In[40]:


start_price = 830.09

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Google')


# In[41]:


# For Amazon Stock - AMZN
AMZN.head()


# In[42]:


start_price = 824.95

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Amazon')


# In[43]:


# Para las acciones de Apple - AAPL
AAPL.head()


# In[44]:


start_price = 117.10

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Analysis "Monte Carlo"  para Apple')


# In[45]:


# Para las acciones de Microsoft - MSFT
MSFT.head()


# In[46]:


start_price = 59.94

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Microsoft')


# Sigamos adelante y obtengamos un histograma de los resultados finales para una corrida mucho mayor. (nota: Esto podría tardar un poco en ejecutarse , dependiendo del número de ejecuciones elegidas)

# In[47]:


# Empecemos por el precio de las acciones de Google
start_price = 830.09

#  Establecer un gran número de ejecuciones
runs = 10000

# Crear una matriz vacía para contener los datos del precio final
simulations = np.zeros(runs)

for run in range(runs):
    # Establezca el punto de datos de la simulación como el último precio de las acciones para esa ejecución
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# Ahora que tenemos nuestra matriz de simulaciones, podemos seguir adelante y trazar un histograma, así como utilizar qunatile para definir nuestro riesgo para esta acción.
#
# Para más información sobre los cuantiles, consulte este enlace: http://en.wikipedia.org/wiki/Quantile

# In[48]:


# Ahora definiremos q como el cuantil empírico del 1%, esto significa básicamente que el 99% de los valores deben caer entre aquí
q = np.percentile(simulations, 1)

# Ahora vamos a trazar la distribución de los precios finales
plt.hist(simulations, bins=200)

# Usando plt.figtext para rellenar alguna información adicional en el gráfico

# Precio inicial
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)

# precio medio final
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())

# Varianza del precio (con un intervalo de confianza del 99%)
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# Para mostrar el cuantil del 1%
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Trazar una línea en el resultado del cuantil del 1%
plt.axvline(x=q, linewidth=4, color='r')

# Para el título de la parcela
plt.title("Final price distribution for Google Stock(GOOGL) after 365 days",
          weight='bold', color='y')


# Ahora hemos observado el cuantil empírico del 1% de la distribución final de precios para estimar el valor en riesgo de las acciones de Google (GOOGL), que parece ser de 17,98 dólares por cada inversión de
# 830,09 (El precio de una acción inicial de Google).
#
# Esto significa básicamente que por cada acción inicial de GOOGL que usted compra está poniendo en riesgo alrededor de 17,98 dólares el 99% del tiempo de nuestra simulación de Monte Carlo.
#

# ##### Ahora vamos a trazar las acciones restantes para estimar el VaR con nuestra simulación de Monte Carlo.

# In[49]:


# Para el precio de las acciones de Amazon
start_price = 824.95

# Establecer un gran número de ejecuciones
runs = 10000

# Crear una matriz vacía para contener los datos del precio final
simulations = np.zeros(runs)

for run in range(runs):
    # Establecer el punto de datos de la simulación como el último precio de las acciones para esa ejecución
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[50]:


# Ahora definiremos q como el cuantil empírico del 1%, esto significa básicamente que el 99% de los valores deben caer entre aquí
q = np.percentile(simulations, 1)

# Ahora vamos a trazar la distribución de los precios finales
plt.hist(simulations, bins=200)

# Usando plt.figtext para rellenar alguna información adicional en el gráfico

# Precio inicial
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)

#  precio medio final
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())

# Varianza del precio (con un intervalo de confianza del 99%)
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# Para mostrar el cuantil del 1%
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Trazar una línea en el resultado del cuantil del 1%
plt.axvline(x=q, linewidth=4, color='r')

# Para el título de la parcela
plt.title(s="Final price distribution for Amazon Stock(AMZN) after %s days" %
          days, weight='bold', color='G')


# Esto significa básicamente que por cada acción inicial de AMZN que compres estarás poniendo en riesgo unos 18,13 dólares el 99% de las veces de nuestra Simulación de Montecarlo.

# In[51]:


# Para el precio de las acciones de Apple
start_price = 117.10

# Establecer un gran número de ejecuciones
runs = 10000

# Crear una matriz vacía para contener los datos del precio final
simulations = np.zeros(runs)

for run in range(runs):
    # Establezca el punto de datos de la simulación como el último precio de las acciones para esa ejecución
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[52]:


# Ahora definiremos q como el cuantil empírico del 1%, esto significa básicamente que el 99% de los valores deben caer entre aquí
q = np.percentile(simulations, 1)

# Ahora vamos a trazar la distribución de los precios finales
plt.hist(simulations, bins=200)

# Usando plt.figtext para rellenar alguna información adicional en el gráfico

# Precio inicial
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)

# precio medio final
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())

# Varianza del precio (con un intervalo de confianza del 99%)
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# Para mostrar el cuantil del 1%
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Trazar una línea en el resultado del cuantil del 1%
plt.axvline(x=q, linewidth=4, color='r')

# Para el título de la parcela
plt.title(s="Final price distribution for Apple Stock(AAPL) after %s days" %
          days, weight='bold', color='B')


# Esto significa básicamente que por cada acción inicial de AAPL que usted compra está poniendo alrededor de 2,48 dólares en riesgo el 99% del tiempo de nuestra simulación de Monte Carlo.

# In[53]:


# Para el precio de las acciones de Microsoft
start_price = 59.94

# Establecer un gran número de ejecuciones
runs = 10000

# Crear una matriz vacía para contener los datos del precio final
simulations = np.zeros(runs)

for run in range(runs):
    # Establezca el punto de datos de la simulación como el último precio de las acciones para esa ejecución
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[54]:


# Ahora definiremos q como el cuantil empírico del 1%, esto significa básicamente que el 99% de los valores deben caer entre aquí
q = np.percentile(simulations, 1)

#  Ahora vamos a trazar la distribución de los precios finales
plt.hist(simulations, bins=200)

# Usando plt.figtext para rellenar alguna información adicional en el gráfico

# Precio inicial
plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)

# precio medio final
plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())

# Varianza del precio (con un intervalo de confianza del 99%)
plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))

# Para mostrar el cuantil del 1%
plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

# Trazar una línea en el resultado del cuantil del 1%
plt.axvline(x=q, linewidth=4, color='r')

# Para el título de la parcela
plt.title(s="Final price distribution for Microsoft Stock(MSFT) after %s days" %
          days, weight='bold', color='M')


# Bien, Esto significa básicamente que por cada acción inicial de MSFT que usted compra está poniendo alrededor de 1,28 dólares en riesgo el 99% del tiempo de nuestra simulación de Monte Carlo.

# Ahora vamos a estimar el Valor en Riesgo (VaR) para una acción relacionada con otros dominios.
#
#
# Estimaremos el VaR para:
# - Johnson & Johnson > JNJ (U.S.: NYSE) [JNJ](http://quotes.wsj.com/JNJ)
# - Wal-Mart Stores Inc. > WMT (U.S.: NYSE) [WMT](http://quotes.wsj.com/WMT)
# - Nike Inc. > NKE (U.S.: NYSE) [NKE](http://quotes.wsj.com/NKE)
#
#
# Utilizando los métodos anteriores para obtener el Valor en Riesgo.

# In[55]:


# Lista de valores de la Bolsa de Nueva York para el análisis
NYSE_list = ['JNJ', 'NKE', 'WMT']

# establecer la hora de inicio y finalización de la toma de datos
end = datetime.now()
start = datetime(end.year-1, end.month, end.day)

# For-loop para agarrar los datos de google finance y establecerlos como dataframe
# Establecer DataFrame como el Stock Ticker

for stock in NYSE_list:
    globals()[stock] = DataReader(stock, 'google', start, end)


# Sigamos jugando con el DataFrame de la acción JNJ(Johnson & Johnson) para conocer los datos.

# In[56]:


JNJ.head()


# In[57]:


JNJ.describe()


# In[58]:


JNJ.info()


# Ahora que hemos visto el DataFrame, vamos a trazar los precios de cierre de las acciones del NYSE.

# In[59]:


# Veamos una vista histórica del precio de cierre de JNJ(Johnson & Johnson)
JNJ['Close'].plot(title='Closing Price - JNJ', legend=True, figsize=(10, 4))


# In[60]:


# Veamos una vista histórica del precio de cierre de NKE(Nike Inc.)
NKE['Close'].plot(title='Closing Price - NKE', legend=True, figsize=(10, 4))


# In[61]:


# Veamos una vista histórica del precio de cierre de WMT(Wal-Mart Stores Inc.)
WMT['Close'].plot(title='Closing Price - WMT', legend=True, figsize=(10, 4))


# ### Valor en riesgo utilizando el método "Bootstrap
#
# Calcularemos los cuantiles empíricos a partir de un histograma de rendimientos diarios.

# Vamos a utilizar pandas para recuperar los rendimientos diarios de las acciones JNJ, WMT y NKE.

# In[62]:


# Utilizaremos pct_change para encontrar el cambio porcentual de cada día

# Para las acciones de JNJ
JNJ['Daily Return'] = JNJ['Close'].pct_change()


# In[63]:


# Tenga en cuenta el uso de dropna() aquí, de lo contrario los valores NaN no pueden ser leídos por seaborn
sns.distplot(JNJ['Daily Return'].dropna(), bins=100, color='R')


# In[64]:


(JNJ['Daily Return'].dropna()).quantile(0.05)


# El cuantil empírico de 0,05 de los rendimientos diarios de las acciones de JNJ está en -0,010. Esto significa que, con un 95% de confianza, nuestra peor pérdida diaria no superará el 1%. Si tenemos una inversión de 1 millón de dólares, nuestro VaR diario del 5% es de 0,010 * 1.000.000 = 10.000 dólares.

# In[65]:


# Para las acciones de WMT
WMT['Daily Return'] = WMT['Close'].pct_change()


# In[66]:


sns.distplot(WMT['Daily Return'].dropna(), bins=100, color='G')


# In[67]:


(WMT['Daily Return'].dropna()).quantile(0.05)


# El cuantil empírico 0,05 de los rendimientos diarios de las acciones de WMT está en -0,013. Esto significa que, con un 95% de confianza, nuestra peor pérdida diaria no superará el 1,3%. Si tenemos una inversión de 1 millón de dólares, nuestro VaR diario del 5% es de 0,013 * 1.000.000 = 13.000 dólares.

# In[68]:


# Para las acciones de NKE
NKE['Daily Return'] = NKE['Close'].pct_change()


# In[69]:


sns.distplot(NKE['Daily Return'].dropna(), bins=100, color='B')


# In[70]:


(NKE['Daily Return'].dropna()).quantile(0.05)


# El cuantil empírico 0,05 de los rendimientos diarios de las acciones de NKE está en -0,018. Esto significa que, con un 95% de confianza, nuestra peor pérdida diaria no superará el 1,8%. Si tenemos una inversión de 1 millón de dólares, nuestro VaR diario del 5% es de 0,018 * 1.000.000 = 18.000 dólares.
# ----------------------------------------------------

 ### Preguntas
#
# En este análisis, me gustaría explorar las siguientes preguntas.
#
# 1. ¿Cuál fue el cambio en el precio de las acciones a lo largo del tiempo?
# 2. ¿Cuál fue el rendimiento diario de la acción en promedio?
# 3. ¿Cuál fue la media móvil de las distintas acciones?
# 4. ¿Cuál fue la correlación entre los precios de cierre de las distintas acciones?
# 4. ¿Cuál es la correlación entre los rendimientos diarios de las distintas acciones?
# 5. ¿Qué valor ponemos en riesgo al invertir en una determinada acción?
# 6. ¿Cómo podemos intentar predecir el comportamiento futuro de las acciones?
