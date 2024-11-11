import streamlit as st
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, kstest, normaltest
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import entropy
from statsmodels.stats.weightstats import ztest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import mord as m
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import eig




st.set_page_config(layout="wide")

# Cargar el archivo CSV
df = pd.read_csv("WineQT.csv", delimiter=",")
df.drop(columns=["Id"], inplace=True)
df = df.replace({',': '.'}, regex=True).apply(pd.to_numeric, errors='coerce')
df["QualityCat"] = ["Alta" if quality >= 7 else "Baja" for quality in df["quality"]]

page = st.sidebar.selectbox("proceso: ", ["EDA","Regresion","Clasificacion"],)

# ---------------------- Funciones ----------------------

# Funcion de RMSE MSE
def evaluate_rmse_mse(y_true, y_pred):
        """Calcula el MSE y el RMSE dados los valores reales y los predichos.
        
        Args:
            y_true: Valores reales (observaciones).
            y_pred: Valores predichos por el modelo.
            
        Returns:
            Tuple: (MSE, RMSE)
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return mse, rmse

# Valores atipicos Total
def contar_valores_atipicos(df):
    # Filtrar las columnas numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Inicializar el contador de valores atípicos
    total_valores_atipicos = 0
    
    # Calcular los valores atípicos en cada columna numérica
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calcular los límites para valores atípicos
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Contar los valores atípicos en la columna
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
        total_valores_atipicos += len(outliers)
    
    return total_valores_atipicos

# Funciones de EDA

def normality_and_statistics(data, column):
    # Verificar que la columna sea de tipo numérico
    if not pd.api.types.is_numeric_dtype(data[column]):
        return f"La columna '{column}' no es de tipo numérico."
    
    # Eliminar valores nulos de la columna para el análisis
    col_data = data[column].dropna()
    
    # Pruebas de normalidad
    shapiro_stat, shapiro_p = shapiro(col_data)
    ks_stat, ks_p = kstest(col_data, 'norm')
    dagostino_stat, dagostino_p = normaltest(col_data)
    
    # Estadísticas descriptivas
    mean = col_data.mean()
    median = col_data.median()
    variance = col_data.var()
    std_dev = col_data.std()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)].count()
    null_values = data[column].isnull().sum()
    
    # Resultados de las pruebas de normalidad
    normality_results = {
        'Prueba': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', "D'Agostino"],
        'Estadístico': [shapiro_stat, ks_stat, dagostino_stat],
        'p-valor': [shapiro_p, ks_p, dagostino_p]
    }
    normality_df = pd.DataFrame(normality_results)
    
    # Estadísticas adicionales
    stats_results = {
        'Media': [mean],
        'Mediana': [median],
        'Varianza': [variance],
        'Desviación': [std_dev],
        #'Q1': [q1],
        #'Q3': [q3],
        #'IQR': [iqr],
        'Nulos': [null_values],
        'Atípicos': [outliers]
    }
    stats_df = pd.DataFrame(stats_results)
    
    return normality_df, stats_df

# QQ-Plot con Plotly
def plot_qqplot(data, column):
    if pd.api.types.is_numeric_dtype(data[column]):
        col_data = data[column].dropna()
        (osm, osr), (slope, intercept, r) = stats.probplot(col_data, dist="norm", plot=None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Datos observados'))
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Línea de referencia'))

        fig.update_layout(
            title=f"QQPlot para{column}",
            xaxis_title='Cuantiles Teóricos',
            yaxis_title='Cuantiles Observados',
            height=400,
            width=600
        )

        st.plotly_chart(fig)
    else:
        st.write(f"La columna '{column}' no es de tipo numérico.")


# Scatter Plot entre dos variables con Plotly, con opción para una variable categórica y colores específicos
def scatter_plot_between_variables(data, var1, var2, category_var=None):
    if var1 in data.columns and var2 in data.columns:
        # Definir secuencia de colores en blanco y rojo
        color_sequence = ['#FFFFFF', '#FF0000'] if category_var in data.columns else None
        
        # Crear el scatter plot con opción de categorizar por color
        fig = px.scatter(
            data_frame=data,
            x=var1,
            y=var2,
            color=category_var if category_var in data.columns else None,
            color_discrete_sequence=color_sequence,
            opacity=0.5,
            title=f'Scatter Plot: {var1} vs {var2}',
            labels={var1: var1, var2: var2, category_var: category_var}
        )
        
        # Actualizar layout
        fig.update_layout(
            title=f'Scatter Plot entre {var1} y {var2}',
            xaxis_title=var1,
            yaxis_title=var2,
        )
        
        st.plotly_chart(fig)
    else:
        st.write("Las variables proporcionadas no están en el dataframe.")

def hypothesis_tests(data, numeric_var, categorical_var):
    # Verificar que las variables están en el DataFrame
    if numeric_var not in data.columns or categorical_var not in data.columns:
        raise ValueError("Las variables proporcionadas no están en el DataFrame.")

    # Extraer las categorías únicas
    categories = data[categorical_var].dropna().unique()
    num_categories = len(categories)

    results = []  # Lista para almacenar los resultados de cada prueba
    
    # Si la variable categórica tiene solo dos categorías
    if num_categories == 2:
        # Dividir el DataFrame en dos grupos
        group1 = data[data[categorical_var] == categories[0]][numeric_var].dropna()
        group2 = data[data[categorical_var] == categories[1]][numeric_var].dropna()
        
        # Prueba Z
        z_stat, z_p_value = ztest(group1, group2)
        results.append({'Prueba': 'Prueba Z', 'Estadístico': z_stat, 'Valor p': z_p_value})
        
        # Prueba T
        t_stat, t_p_value = stats.ttest_ind(group1, group2)
        results.append({'Prueba': 'Prueba T', 'Estadístico': t_stat, 'Valor p': t_p_value})
    
    # Si la variable categórica tiene tres o más categorías
    elif num_categories >= 3:
        # Crear lista de grupos para las pruebas ANOVA y Kruskal-Wallis
        groups = [data[data[categorical_var] == category][numeric_var].dropna() for category in categories]
        
        # Prueba ANOVA
        f_stat, f_p_value = stats.f_oneway(*groups)
        results.append({'Prueba': 'ANOVA', 'Estadístico': f_stat, 'Valor p': f_p_value})
        
        # Prueba de Kruskal-Wallis
        kruskal_stat, kruskal_p_value = stats.kruskal(*groups)
        results.append({'Prueba': 'Kruskal-Wallis', 'Estadístico': kruskal_stat, 'Valor p': kruskal_p_value})
    
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Función para crear un boxplot basado en una variable numérica y una categórica
def boxplot_by_category(data, numeric_var, categorical_var):
    # Verificar que ambas variables estén en el DataFrame
    if numeric_var in data.columns and categorical_var in data.columns:
        # Crear el gráfico de boxplot
        fig = px.box(
            data_frame=data,
            x=categorical_var,
            y=numeric_var,
            points="all",  # Muestra todos los puntos para mayor detalle
            title=f'Boxplot de {numeric_var} por {categorical_var}',
            labels={categorical_var: categorical_var, numeric_var: numeric_var}
        )
        
        # Personalizar el diseño
        fig.update_layout(
            xaxis_title=categorical_var,
            yaxis_title=numeric_var,
            boxmode="group"  # Agrupa los boxplots según las categorías
        )
        
        return fig
    else:
        raise ValueError("Las variables proporcionadas no están en el DataFrame o no tienen el tipo correcto.")


# Correlaciones entre dos variables
def correlation_between_two(data, var1, var2):
    if pd.api.types.is_numeric_dtype(data[var1]) and pd.api.types.is_numeric_dtype(data[var2]):
        pearson_corr = data[var1].corr(data[var2], method='pearson')
        kendall_corr = data[var1].corr(data[var2], method='kendall')
        spearman_corr = data[var1].corr(data[var2], method='spearman')
        
        result = {
            'Correlación': ['Pearson', 'Kendall', 'Spearman'],
            'Valor': [pearson_corr, kendall_corr, spearman_corr]
        }
        
        return pd.DataFrame(result)
    else:
        return f"Ambas variables '{var1}' y '{var2}' deben ser numéricas."

# Función para graficar la matriz de correlación
def plot_correlation_matrix(data, method='pearson'):
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("El método debe ser 'pearson', 'kendall' o 'spearman'.")
    
    # Calcular la matriz de correlación en función del método seleccionado
    corr_matrix = data.corr(method=method)
    
    # Crear el heatmap con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, 
        x=corr_matrix.columns, 
        y=corr_matrix.columns, 
        colorscale='RdBu', 
        zmin=-1, 
        zmax=1,
        colorbar=dict(title="Correlación"),
        hoverongaps=False
    ))

    # Añadir los valores numéricos en los recuadros
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(dict(
                x=corr_matrix.columns[i],
                y=corr_matrix.columns[j],
                text=str(round(corr_matrix.iloc[j, i], 2)),
                showarrow=False,
                font=dict(color="black")
            ))

    # Actualizar el layout para mejorar el diseño
    fig.update_layout(
        title=f'Matriz de Correlación ({method.capitalize()})',
        xaxis_nticks=36,
        width=800, 
        height=700,
        yaxis_autorange='reversed'
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

def plot_bars_and_entropy(df, column):
    # Calcular la frecuencia de cada valor en la variable seleccionada
    value_counts = df[column].value_counts()
    
    # Crear el gráfico de barras con Plotly
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': column, 'y': 'Frecuencia'},
        title=f'Distribución de {column}'
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Frecuencia", height=500)
    
    return fig

def plot_boxplot(df, variable):
    """
    Genera un boxplot de una variable específica de un DataFrame usando Plotly.

    Parameters:
    df (pd.DataFrame): DataFrame que contiene los datos.
    variable (str): Nombre de la columna para la cual se desea crear el boxplot.

    Returns:
    plotly.graph_objs._figure.Figure: Figura del boxplot.
    """
    # Verifica si la variable está en el DataFrame
    if variable not in df.columns:
        raise ValueError(f"La variable '{variable}' no se encuentra en el DataFrame")

    # Genera el boxplot usando Plotly Express
    fig = px.box(df, y=variable, title=f"Boxplot de {variable}")
    fig.update_layout(yaxis_title=variable, height=500)
    
    return fig

def plot_histogram(df, variable, nbins=20):
    """
    Genera un histograma de una variable específica de un DataFrame usando Plotly.

    Parameters:
    df (pd.DataFrame): DataFrame que contiene los datos.
    variable (str): Nombre de la columna para la cual se desea crear el histograma.
    nbins (int): Número de bins para el histograma (por defecto es 20).

    Returns:
    plotly.graph_objs._figure.Figure: Figura del histograma.
    """
    # Verificar si la variable está en el DataFrame
    if variable not in df.columns:
        raise ValueError(f"La variable '{variable}' no se encuentra en el DataFrame")
    
    # Generar el histograma usando Plotly Express
    fig = px.histogram(df, x=variable, nbins=nbins, title=f"Histograma de {variable}")
    fig.update_layout(xaxis_title=variable, yaxis_title="Frecuencia", height=500)
    
    return fig

def imputar_datos(df, tipo_imputacion="null", metodo="media"):
    df_imputado = df.copy()
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns

    if tipo_imputacion == "null":
        # Imputar valores nulos
        if metodo == "media":
            df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(df[columnas_numericas].mean())
        elif metodo == "mediana":
            df_imputado[columnas_numericas] = df_imputado[columnas_numericas].fillna(df[columnas_numericas].median())
        elif metodo == "iterativa":
            imputer = IterativeImputer()
            df_imputado[columnas_numericas] = imputer.fit_transform(df_imputado[columnas_numericas])

    elif tipo_imputacion == "atipicos":
        # Imputar valores atípicos
        for col in columnas_numericas:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            if metodo == "media":
                media = df[col].mean()
                df_imputado.loc[(df[col] < limite_inferior) | (df[col] > limite_superior), col] = media
            elif metodo == "mediana":
                mediana = df[col].median()
                df_imputado.loc[(df[col] < limite_inferior) | (df[col] > limite_superior), col] = mediana
            elif metodo == "iterativa":
                imputer = IterativeImputer()
                df_imputado[columnas_numericas] = imputer.fit_transform(df_imputado[columnas_numericas])
                break  # Iterative imputación para todo el DataFrame (sin límite específico)
    
    return df_imputado

def eliminar_columnas(df, columnas_a_eliminar):
    """
    Elimina las columnas especificadas de un DataFrame.

    Parámetros:
    - df: DataFrame original.
    - columnas_a_eliminar: Lista de nombres de columnas a eliminar.

    Retorna:
    - Un nuevo DataFrame sin las columnas especificadas.
    """
    df_nuevo = df.drop(columns=columnas_a_eliminar, errors='ignore')
    return df_nuevo






def plot_residuals(residuals):
    # Calcular la media de los residuos
    residuals_mean = np.mean(residuals)
    
    # Crear el histograma
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        name='Residuos'
    ))

    # Agregar la línea de la media
    fig.add_trace(go.Scatter(
        x=[residuals_mean, residuals_mean],
        y=[0, max(np.histogram(residuals, bins=30)[0])],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'Media: {residuals_mean:.2f}'
    ))

    # Agregar título y etiquetas
    fig.update_layout(
        title="Distribución de los Residuos",
        xaxis_title="Residuos",
        yaxis_title="Frecuencia",
        showlegend=True,
        width = 500,
        height= 500
    )

    # Mostrar el gráfico
    return fig


def plot_residuals_vs_predictions(predictions, residuals):
    # Calcular la media de los residuos
    residuals_mean = np.mean(residuals)

    # Crear el gráfico de dispersión
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        name='Residuos'
    ))

    # Agregar la línea de la media de los residuos
    fig.add_trace(go.Scatter(
        x=[min(predictions), max(predictions)],
        y=[residuals_mean, residuals_mean],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'Media: {residuals_mean:.2f}'
    ))

    # Ajustar el layout
    fig.update_layout(
        title="Residuos vs Valores Predichos (Supuesto de Media Cero)",
        xaxis_title="Valores predichos",
        yaxis_title="Residuos",
        width=500,
        height=500,
        showlegend=True
    )

    # Mostrar el gráfico
    return fig


def homoscedasticity_tests(residuals, data):
    # Añadir constante a los datos
    X = sm.add_constant(data)

    # Realizar la prueba de Breusch-Pagan
    bp_test = sms.het_breuschpagan(residuals, X)
    bp_stat, bp_pval = bp_test[0], bp_test[1]

    # Realizar la prueba de White
    white_test = sms.het_white(residuals, X)
    white_stat, white_pval = white_test[0], white_test[1]

    # Realizar la prueba de Goldfeld-Quandt
    gq_test = sms.het_goldfeldquandt(residuals, X)
    gq_stat, gq_pval = gq_test[0], gq_test[1]

    # Crear el DataFrame con los resultados
    results_df = pd.DataFrame({
        'Prueba': ['Breusch-Pagan', 'White', 'Goldfeld-Quandt'],
        'Estadístico': [bp_stat, white_stat, gq_stat],
        'Valor p': [bp_pval, white_pval, gq_pval]
    })

    return results_df

def plot_residuals_by_index(residuals):
    # Crear el gráfico de dispersión
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))),
        y=residuals,
        mode='markers',
        marker=dict(color='skyblue', opacity=0.6, line=dict(color='black', width=1)),
        name='Residuos'
    ))

    # Agregar la línea horizontal en y=0
    fig.add_trace(go.Scatter(
        x=[0, len(residuals)-1],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Línea en y=0'
    ))

    # Ajustar layout
    fig.update_layout(
        title="Residuos en función del índice de observación (Supuesto de Independencia)",
        xaxis_title="Índice de observación",
        yaxis_title="Residuos",
        showlegend=True
    )

    # Mostrar el gráfico
    return fig

def durbin_watson_test(residuals):
    # Calcular el estadístico de Durbin-Watson
    dw_stat = sm.stats.durbin_watson(residuals)

    # Crear un DataFrame con el resultado
    results_df = pd.DataFrame({
        'Prueba': ['Durbin-Watson'],
        'Estadístico': [dw_stat]
    })

    return results_df


def qq_plot(residuals):
    # Calcular los valores teóricos y observados para el gráfico Q-Q
    qq = stats.probplot(residuals, dist="norm")
    theoretical_quantiles = qq[0][0]
    sample_quantiles = qq[0][1]

    # Crear el gráfico Q-Q
    fig = go.Figure()

    # Puntos Q-Q
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Quantiles'
    ))

    # Línea de referencia (y = x)
    min_q = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_q = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[min_q, max_q],
        y=[min_q, max_q],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Línea de referencia (y = x)'
    ))

    # Configurar el layout
    fig.update_layout(
        title="Gráfico Q-Q de los residuos (Supuesto de Normalidad)",
        xaxis_title="Cuantiles teóricos",
        yaxis_title="Cuantiles muestrales",
        showlegend=True
    )

    # Mostrar el gráfico
    return fig


def normality_tests(col_data):

    shapiro_stat, shapiro_p = shapiro(col_data)
    ks_stat, ks_p = kstest(col_data, 'norm')
    dagostino_stat, dagostino_p = normaltest(col_data)
    

    normality_results = {
        'Prueba': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', "D'Agostino"],
        'Estadístico': [shapiro_stat, ks_stat, dagostino_stat],
        'p-valor': [shapiro_p, ks_p, dagostino_p]
    }
    normality_df = pd.DataFrame(normality_results)

    return normality_df

def calculate_vif(data):
    # Crear un DataFrame para almacenar el VIF de cada variable
    vif_data = pd.DataFrame()
    vif_data['Variable'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    
    return vif_data


def eigenvalue_analysis(data):
    # Calcular X^T X
    xtx = np.dot(data.T, data)
    
    # Calcular los valores propios de X^T X
    eigenvalues, _ = eig(xtx)
    
    # Calcular la dispersión espectral
    kdisp = (max(eigenvalues) / min(eigenvalues)) ** (1/2)
    
    # Mostrar resultados en Streamlit
    st.write("##### Análisis de los valores propios de X^T X")
    st.write("La dispersión en el espectro de los valores propios de la matriz es:", kdisp)

    return kdisp


def check_list(residuals, data):
    # 1. Supuesto de media cero en residuos
    residuals_mean = residuals.mean()
    mean_check = "✔️ Media cercana a 0" if abs(residuals_mean) < 0.05 else "❌ Media alejada de 0"
    
    # 2. Supuesto de normalidad
    normality_df = normality_tests(residuals)
    normality_check = "✔️ Dos o más pruebas indican normalidad" if len(normality_df[normality_df['p-valor'] > 0.05]) >= 2 else "❌ Menos de dos pruebas indican normalidad"
    
    # 3. Supuesto de independencia (Durbin-Watson)
    dw_df = durbin_watson_test(residuals)
    dw_check = "✔️ Residuos independientes" if dw_df['Estadístico'][0] < 2.5 else "❌ Posible autocorrelación en residuos"
    
    # 4. Supuesto de homocedasticidad
    homoscedasticity_df = homoscedasticity_tests(residuals, X_train_scaled)
    homoscedasticity_check = "✔️ Dos o más pruebas indican homocedasticidad" if len(homoscedasticity_df[homoscedasticity_df['Valor p'] > 0.05]) >= 2 else "❌ Menos de dos pruebas indican homocedasticidad"
    
    # 5. Supuesto de multicolinealidad (VIF)
    vif_df = calculate_vif(data)
    multicollinearity_check = "✔️ No hay multicolinealidad" if all(vif_df['VIF'] < 10) else "❌ Posible multicolinealidad detectada"
    
    # 6. Dispersión en el espectro de valores propios (análisis espectral)
    kdisp = eigenvalue_analysis(data)
    spectral_dispersion_check = "✔️ Sin problemas de dispersión espectral" if kdisp < 30 else "❌ Alta dispersión espectral"

    # Mostrar el checklist en Streamlit
    st.subheader("Resumen de los Supuestos del Modelo de Regresión")
    st.write("#### 1. Supuesto de Media Cero en los Residuos")
    st.write(mean_check)
    
    st.write("#### 2. Supuesto de Normalidad de los Residuos")
    st.write(normality_check)
    
    st.write("#### 3. Supuesto de Independencia de los Residuos (Durbin-Watson)")
    st.write(dw_check)
    
    st.write("#### 4. Supuesto de Homocedasticidad de los Residuos")
    st.write(homoscedasticity_check)
    
    st.write("#### 5. Supuesto de Ausencia de Multicolinealidad (VIF)")
    st.write(multicollinearity_check)
    
    st.write("#### 6. Supuesto de Ausencia de Dispersión Espectral")
    st.write(spectral_dispersion_check)


# Opciones de análisis
if page == "EDA":

    st.title("Analisis Exploratorio de Datos (EDA)")

    st.header("Variables del Modelo")
    
    content = f"""
        La base de datos **"Wine Quality Dataset"** es un conjunto de datos utilizado para analizar y predecir la calidad del vino basándose en distintas características físico-químicas del producto. Esta base incluye variables que representan atributos clave en la composición del vino, así como una variable de salida que califica su calidad de acuerdo con una evaluación sensorial. A continuación, se presenta una descripción detallada de cada variable:

        - **Fixed Acidity (Acidez fija):** mide los ácidos no volátiles, como el tartárico. La acidez fija es importante para la estabilidad y frescura del vino.
        - **Volatile Acidity (Acidez volátil):** corresponde a los ácidos que pueden evaporarse, principalmente el ácido acético, que en altos niveles puede dar un sabor de vinagre.
        - **Citric Acid (Ácido cítrico):** contribuye a mejorar el sabor y la frescura del vino.
        - **Residual Sugar (Azúcar residual):** cantidad de azúcar que queda después de la fermentación. Niveles más altos hacen que el vino sea más dulce.
        - **Chlorides (Cloruros):** cantidad de sales, que afecta directamente el sabor y la calidad del vino.
        - **Free Sulfur Dioxide (Dióxido de azufre libre):** SO₂ en forma activa que previene el crecimiento microbiano y la oxidación del vino.
        - **Total Sulfur Dioxide (Dióxido de azufre total):** cantidad total de SO₂ en el vino, que incluye tanto el libre como el ligado.
        - **Density (Densidad):** influye en la percepción del cuerpo del vino, y está relacionada con el contenido de alcohol y azúcares.
        - **pH:** mide la acidez o basicidad del vino, influyendo en su estabilidad y percepción de sabor.
        - **Sulphates (Sulfatos):** contribuyen a la conservación y antioxidación, además de mejorar el sabor.
        - **Alcohol:** porcentaje de contenido alcohólico, influye en el sabor y cuerpo del vino.
        - **Quality (Calidad):** variable de salida que mide la calidad del vino mediante una puntuación sensorial en una escala de 0 a 10.
        - **QualityCat (Calidad):** variable categorica de salida que mide la calidad del vino con base en la variable Quality, la categoria 0 [0 - 6], categoria 1 [7 - 10].

        Numero de Registros:    {len(df)}

        Numero Valores Faltantes: {df.isnull().sum().sum()}

        Numero valores Atipicos: {contar_valores_atipicos(df)}
    """

    st.markdown(content)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader('Tipo de Análisis')
        analysis_type = st.selectbox('Selecciona el tipo de análisis:', ['Univariado', 'Bivariado', 'Multivariado','Conclusiones'])

    if analysis_type == 'Univariado':
        
        with col2:
            
            st.subheader('Variables')
            columns = df.columns.tolist() 
            selected_column = st.selectbox('Selecciona una variable para el análisis univariado', columns)
            
    elif analysis_type == 'Bivariado':

        with col2:
            st.subheader('Variables')
            col3, col4 = st.columns(2)
            columns = df.columns.tolist() 

            with col3:
                selected_var1 = st.selectbox('Selecciona la primera variable', columns)
            with col4:
                selected_var2 = st.selectbox('Selecciona la segunda variable', columns)

    if analysis_type == 'Univariado':
        
        col1, col2  = st.columns(2)
                
        if selected_column != "QualityCat":  # Si es una variable numérica

                with col1:
                    plot_qqplot(df, selected_column)

                    st.plotly_chart(plot_histogram(df, selected_column))


                with col2:

                    normality_result, statistics = normality_and_statistics(df, selected_column)
                    st.markdown(f'<h5> </h5>', unsafe_allow_html=True)
                    st.markdown(f'<h5>Información {selected_column}</h5>', unsafe_allow_html=True)
                    
                    st.dataframe(normality_result.reset_index(drop=True), use_container_width=True)

                    # Evaluar los resultados de las pruebas de normalidad
                    p_values = normality_result['p-valor']
                    normal_count = sum(p > 0.05 for p in p_values)  # Cuenta las pruebas que indican normalidad
                    

                    # Indicador de normalidad
                    if normal_count >= 2:
                        st.success("Distribución Normal: ✔️", icon="✅")  # Indicador verde
                    else:
                        st.error("Distribución No Normal: ❌", icon="🚫")  # Indicador rojo

                    st.dataframe(statistics.reset_index(drop=True), use_container_width=True)

                    st.plotly_chart(plot_boxplot(df, selected_column))

        else:  # Si es una variable categórica

                graph = plot_bars_and_entropy(df, selected_column)

                with col1:
                    
                    st.subheader(f'Gráfico de Barras y Entropía para {selected_column}')
                    st.plotly_chart(graph)

    elif analysis_type == 'Bivariado':

        col1, col2 = st.columns(2)

        if selected_var2 == "QualityCat":

            with col2:
                st.markdown(f'<h3>Correlacion entre {selected_var1} y {selected_var2}</h3>', unsafe_allow_html=True)

                correlation_result = hypothesis_tests(df, selected_var1, selected_var2)

                st.dataframe(correlation_result, use_container_width=True)

                correlation_values = correlation_result['Valor p']
                high_corr_count = sum(abs(val) >= 0.8 for val in correlation_values)  # Cuenta los coeficientes altos
                    
                # Indicador de correlación
                if high_corr_count >= 2:
                    st.success("Las variables están correlacionadas: ✔️", icon="✅")  # Indicador verde
                else:
                    st.warning("Las variables no están correlacionadas: ⚠️", icon="⚠️")  # Indicador de advertencia

            with col1:

                st.plotly_chart(boxplot_by_category(df, selected_var1, selected_var2))

        else:

            with col2:
                st.markdown(f'<h3>Correlaciones entre {selected_var1} y {selected_var2}</h3>', unsafe_allow_html=True)
                correlation_result = correlation_between_two(df, selected_var1, selected_var2)

                st.dataframe(correlation_result, use_container_width=True)

                                # Evaluar si las variables están correlacionadas
                correlation_values = correlation_result['Valor']
                high_corr_count = sum(abs(val) >= 0.8 for val in correlation_values)  # Cuenta los coeficientes altos
                    
                # Indicador de correlación
                if high_corr_count >= 2:
                    st.success("Las variables están correlacionadas: ✔️", icon="✅")  # Indicador verde
                else:
                    st.warning("Las variables no están correlacionadas: ⚠️", icon="⚠️")  # Indicador de advertencia

            with col1:

                scatter_plot_between_variables(df, selected_var1, selected_var2,"QualityCat")

    elif analysis_type == 'Multivariado':

        col1, col2 = st.columns(2)

        with col1:
            st.header('Análisis Multivariado')
            method = st.selectbox("Selecciona el método de correlación:", ["pearson", "kendall", "spearman"])
            
        # Crear un DataFrame solo con las columnas numéricas
        df_numeric = df.select_dtypes(include=['number'])

        st.subheader(f'Matriz de Correlación ({method.capitalize()})')
        plot_correlation_matrix(df_numeric, method)

    elif analysis_type == 'Conclusiones':

        content = f"""
                Dado que el número de registros es limitado (1143 muestras) y la cantidad de valores atípicos es significativa (437), se procederá a imputar estos valores atípicos utilizando la mediana de los datos en cada variable numérica para reducir su impacto en los modelos.

                Además, para las técnicas de Regresión Logística y KNN, se aplicarán métodos de sobremuestreo en la clase de calidad alta, ya que existe un desbalance severo en las clases. La clase de calidad alta representa solo el 13% del total de muestras, por lo que el sobremuestreo ayudará a mejorar la precisión del modelo al tratar de identificar esta clase minoritaria.

                Finalmente, se realizará una limpieza de variables correlacionadas con un umbral de 0.65 y -0.65, utilizando pruebas de correlación de Pearson, Kendall y Spearman. Este paso permitirá reducir la multicolinealidad entre las variables y mejorar la robustez de los modelos predictivos.
                """
        st.markdown(content)


elif page == "Regresion":

# Ocular codigo

    # Paso 1: Crear un selectbox para seleccionar el modelo
    modelo_seleccionado = st.selectbox("Selecciona el tipo de regresión:", ["Regresión Múltiple", "Regresión Polinómica"])

    # Filtrar y preparar el DataFrame
    dfl = df.drop(columns=["QualityCat", "citric acid", "density", "pH", "total sulfur dioxide"])
    dfl = imputar_datos(dfl, "atipicos", "iterativa")

    # Separar la variable objetivo y las características
    X = dfl.drop(columns=["quality"])
    y = dfl["quality"]

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar la validación cruzada con k = 5
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Normalización y validación cruzada en el conjunto de entrenamiento
    mse_scores = []
    mae_scores = []
    r2_scores = []
    coeficientes = []

    # Configurar el modelo según la selección
    if modelo_seleccionado == "Regresión Múltiple":
        modelo = LinearRegression()
    elif modelo_seleccionado == "Regresión Polinómica":
        modelo = LinearRegression()
        poly = PolynomialFeatures(degree=2)  # Puedes ajustar el grado del polinomio

    # Realizar la validación cruzada
    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Transformar a matriz de NumPy para la validación cruzada
        if modelo_seleccionado == "Regresión Polinómica":
            X_train_fold = poly.fit_transform(X_train_fold)
            X_val_fold = poly.transform(X_val_fold)
        
        # Normalizar
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        # Entrenar el modelo
        modelo.fit(X_train_fold_scaled, y_train_fold)
        
        # Predicciones
        y_val_pred = modelo.predict(X_val_fold_scaled)
        
        # Calcular MSE, MAE y R2
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        r2 = r2_score(y_val_fold, y_val_pred)
        
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        # Guardar coeficientes
        coeficientes.append(modelo.coef_)


    # Calcular el MSE, MAE, R2 promedio
    mse_promedio = np.mean(mse_scores)
    mae_promedio = np.mean(mae_scores)
    r2_promedio = np.mean(r2_scores)

    # Transformar los datos de prueba en el caso de regresión polinómica
    if modelo_seleccionado == "Regresión Polinómica":
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

    # Entrenar el modelo final en el conjunto de entrenamiento completo
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    modelo.fit(X_train_scaled, y_train)

    # Evaluar en el conjunto de prueba
    y_test_pred = modelo.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calcular RMSE
    rmse_test = np.sqrt(mse_test)

    # Calcular R2 ajustado
    n = X_test.shape[0]  # número de muestras
    p = X_test.shape[1]  # número de características
    r2_ajustado = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

    # Crear layout de dos columnas en Streamlit
    col1, col2 = st.columns(2)

    y_train_pred = modelo.predict(X_train_scaled)  # Predicciones para el conjunto de entrenamiento
    residuals = y_train - y_train_pred 
# Ocultar

    # Columna 1: Mostrar las variables y sus coeficientes
    with col1:
        if modelo_seleccionado == "Regresión Polinómica":
            # Para regresión polinómica, los coeficientes corresponden a las nuevas características generadas
            coef_df = pd.DataFrame(modelo.coef_.reshape(1, -1), columns=[f"X{i}" for i in range(1, len(modelo.coef_) + 1)])
        else:
            # Para regresión múltiple, si tiene solo un coeficiente por cada variable, lo asignamos a las columnas de X
            coef_df = pd.DataFrame(modelo.coef_.reshape(1, -1), columns=X.columns)
        
        st.subheader("Coeficientes del modelo")
        st.dataframe(coef_df, use_container_width=True)





    # Columna 2: Mostrar las tablas de validación cruzada y resultados generales
    with col2:
        col3, col4 = st.columns(2)
        with col3:

            # Tabla de resultados generales del modelo
            general_results = {
                "MSE (Prueba)": mse_test,
                "MAE (Prueba)": mae_test,
                "R2 (Prueba)": r2_test,
            }
            general_df = pd.DataFrame(list(general_results.items()), columns=["Métrica", "Valor"])
            st.write("Resultados Generales del Modelo")
            st.dataframe(general_df, use_container_width=True)  

            
        with col4:

            # Tabla de resultados generales del modelo
            kfold_results = {
                "MSE (Prueba)": np.mean(mse_scores),
                "MAE (Prueba)": np.mean(mae_scores),
                "R2 (Prueba)": np.mean(r2_scores),
            }
            kfold_df = pd.DataFrame(list(kfold_results.items()), columns=["Métrica", "Valor"])
            st.write("Resultados kfold del Modelo")
            st.dataframe(kfold_df, use_container_width=True)

    st.subheader("Supuesto Normalidad Residuos y Media = 0")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_residuals(residuals))
    with col2:
        st.plotly_chart(qq_plot(residuals))

    st.dataframe(normality_tests(residuals),use_container_width=True)

    st.subheader("Supuesto Homocedasticidad Residuos")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_residuals_vs_predictions(y_train_pred,residuals))
    with col2:
        st.write(" ")
        st.write(" ")
        st.dataframe(homoscedasticity_tests(residuals, X_train_scaled),use_container_width=True)

    st.subheader("Supuesto Independencia Residuos")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_residuals_by_index(residuals))
    with col2:
        st.write(" ")
        st.write(" ")
        st.dataframe(durbin_watson_test(residuals),use_container_width=True)

    st.subheader("MultiColinealidad")
    
    st.dataframe(calculate_vif(dfl.drop(columns=["quality"])),use_container_width=True)
    

    check_list(residuals, dfl.drop(columns=["quality"]))