from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder 

def entrenarNN1():
	#Leemos la data y convertimos a float los valores de cada columna
	conf = SparkConf().setAppName("NN_1").setMaster("local")
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)
	rdd = sqlContext.read.csv("/home/ulima-azure/data/Enfermedad_Oncologica_T3.csv", header = True).rdd
	rdd = rdd.map(lambda x: (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), 
		float(x[6]), float(x[7]), float(x[8]), float(x[9]))) 

	df = rdd.toDF(["Cellenght", "Cellsize", "Cellshape", "mgadhesion", "sepics", "bnuclei", "bchromatin", "nucleos", "mitoses", 
			"P_Benigno"])
	#Construir nuestro vector assembler (features)
	assembler = VectorAssembler(inputCols=["Cellenght", "Cellsize", "Cellshape", "mgadhesion", "sepics", "bnuclei", 
		"nucleos", "bchromatin", "mitoses"],
		outputCol="features")
	df = assembler.transform(df)

	#Dividir data en training y test
	(df_training, df_test) = df.randomSplit([0.7, 0.3])
	
	#Definir arquitectura de nuestra red (hiperparametro)
	capas = [9, 4, 6, 2]

	#Construimos al entrenador
	#Hiperparametro: maxIter
	entrenador = MultilayerPerceptronClassifier(
		featuresCol="features", labelCol="P_Benigno", maxIter=1000, layers=capas
	)

	#Entrenar nuestro modelo
	modelo = entrenador.fit(df_training)

	#Validar nuestro modelo
	df_predictions = modelo.transform(df_test)
	evaluador = MulticlassClassificationEvaluator(
		labelCol="P_Benigno", predictionCol="prediction", metricName="accuracy"
	)
	accuracy = evaluador.evaluate(df_predictions)
	print(f"Accuracy : {accuracy}")

	df_predictions.select("prediction", "rawPrediction", "probability").show()
	#Mostramos la cantidad de 0 y 1 de las predicciones
	df_predictions.groupby('prediction').count().show()

def clasificar_chi2():
	#Leemos la data y convertimos a float los valores de cada columna
	conf = SparkConf().setAppName("NN_1").setMaster("local")
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)
	rdd = sqlContext.read.csv("/home/ulima-azure/data/Enfermedad_Oncologica_T3.csv", header = True).rdd
	rdd = rdd.map(lambda x: (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]),
		float(x[6]), float(x[7]), float(x[8]), float(x[9])))

	df = rdd.toDF(["Cellenght", "Cellsize", "Cellshape", "mgadhesion", "sepics", "bnuclei", "bchromatin", "nucleos", "mitoses",
                        "P_Benigno"])
	#Construir nuestro vector assembler (features)
	assembler = VectorAssembler(inputCols=["Cellenght", "Cellsize", "Cellshape",
	"nucleos", "bchromatin", "mitoses"],
		outputCol="featuresChi2")
	df_chi2 = assembler.transform(df)
	df_chi2 = df_chi2.select("featuresChi2", "P_Benigno")

	selector = ChiSqSelector(
		numTopFeatures=3,
		featuresCol="featuresChi2",
		labelCol="P_Benigno",
		outputCol="featuresSelected")
	df_result = selector.fit(df_chi2).transform(df_chi2)

	#Dividir data en training y test
	(df_training, df_test) = df_result.randomSplit([0.7, 0.3])

	# Definir arquitectura de nuestra red (hiperparametro)
	capas = [3, 4, 6, 2]

	# Construimos al entrenador
	# Hiperparametro: maxIter
	entrenador = MultilayerPerceptronClassifier(
		featuresCol="featuresSelected", labelCol="P_Benigno", maxIter=1000, layers=capas
	)
	# Entrenar nuestro modelo
	modelo = entrenador.fit(df_training)

	# Validar nuestro modelo
	df_predictions = modelo.transform(df_test)
	evaluador = MulticlassClassificationEvaluator(
		labelCol="P_Benigno", predictionCol="prediction", metricName="accuracy"
	)
	accuracy = evaluador.evaluate(df_predictions)
	print(f"Accuracy: {accuracy}")

	df_predictions.select("prediction", "rawPrediction", "probability").show()
	
	#Mostramos la cantidad de 0 y 1 de las predicciones
	df_predictions.groupby('prediction').count().show()

def main():
	entrenarNN1()
	#clasificar_chi2()

if __name__ == "__main__":
	main()
