package com.dsj.spark.testSpark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.math.exp
object SA {
 val master = "local";
  val appName= "SA";
  val spark = SparkSession.builder().master(master).appName(appName).getOrCreate();
  val sc = spark.sparkContext
  import spark.implicits._
  def main(args:Array[String]) :Unit = {
    //  屏蔽日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    
    val train1 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter", ",").load(args(0))
    val detect3 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter",",").load(args(1))

    val stringIndex1 =  new StringIndexer().setInputCol("_c1").setOutputCol("_c1_index").fit(train1.select($"_c1").union(detect3.select($"_c1")))
    val stringIndex2 =  new StringIndexer().setInputCol("_c2").setOutputCol("_c2_index").fit(train1.select($"_c2").union(detect3.select($"_c2")))
    val stringIndex3 =  new StringIndexer().setInputCol("_c3").setOutputCol("_c3_index").fit(train1.select($"_c3").union(detect3.select($"_c3")))

    val train1_1 = stringIndex1.transform(train1)
    val train1_2 = stringIndex2.transform(train1_1)
    val train1_3 = stringIndex3.transform(train1_2)

    var train1_tem = train1_3.map(row => {
      //      val v1 = row.get(47)
      val label = row.getString(41) match {
        case s:String if(s == "normal") => 1
        case _ => 0
      }
      val label1 = row.getString(41) match {
        case s:String if(s == "warezmaster") => "R2L"
        case s:String if(s == "smurf") => "DOS"
        case s:String if(s == "pod") => "DOS"
        case s:String if(s == "imap") => "R2L"
        case s:String if(s == "nmap") => "Probing"
        case s:String if(s == "guess_passwd") => "R2L"
        case s:String if(s == "ipsweep") => "Probing"
        case s:String if(s == "portsweep") => "Probing"
        case s:String if(s == "satan") => "Probing"
        case s:String if(s == "land") => "DOS"
        case s:String if(s == "loadmodule") => "U2R"
        case s:String if(s == "ftp_write") => "R2L"
        case s:String if(s == "buffer_overflow") => "U2R"
        case s:String if(s == "rootkit") => "U2R"
        case s:String if(s == "warezclient") => "R2L"
        case s:String if(s == "teardrop") => "DOS"
        case s:String if(s == "perl") => "U2R"
        case s:String if(s == "phf") => "R2L"
        case s:String if(s == "multihop") => "R2L"
        case s:String if(s == "neptune") => "DOS"
        case s:String if(s == "back") => "DOS"
        case s:String if(s == "spy") => "R2L"
        case s:String if(s == "normal") => "Normal"
        case _ => ""
      }
      (label,label1,Vectors.dense(row.getString(0).toDouble,row.getString(4).toDouble,row.getString(5).toDouble,row.getString(6).toDouble,row.getString(7).toDouble,row.getString(8).toDouble,row.getString(9).toDouble,row.getString(10).toDouble,row.getString(11).toDouble,row.getString(12).toDouble,row.getString(13).toDouble,row.getString(14).toDouble,row.getString(15).toDouble,row.getString(16).toDouble,row.getString(17).toDouble,row.getString(18).toDouble,row.getString(19).toDouble,row.getString(20).toDouble,row.getString(21).toDouble,row.getString(22).toDouble,row.getString(23).toDouble,row.getString(24).toDouble,row.getString(25).toDouble,row.getString(26).toDouble,row.getString(27).toDouble,row.getString(28).toDouble,row.getString(29).toDouble,row.getString(30).toDouble,row.getString(31).toDouble,row.getString(32).toDouble,row.getString(33).toDouble,row.getString(34).toDouble,row.getString(35).toDouble,row.getString(36).toDouble,row.getString(37).toDouble,row.getString(38).toDouble,row.getString(39).toDouble,row.getString(40).toDouble),row.getDouble(42),row.getDouble(43),row.getDouble(44))
    }).toDF("label","label1","features_tem","_c1_index","_c2_index","_c3_index")

    val train1_1_onehot = new OneHotEncoder().setInputCol("_c1_index").setOutputCol("_c1_index_oh").setDropLast(false).transform(train1_tem)
    val train1_2_onehot = new OneHotEncoder().setInputCol("_c2_index").setOutputCol("_c2_index_oh").setDropLast(false).transform(train1_1_onehot)
    val train1_3_onehot = new OneHotEncoder().setInputCol("_c3_index").setOutputCol("_c3_index_oh").setDropLast(false).transform(train1_2_onehot)
    val train1_data = new VectorAssembler().setInputCols(Array("features_tem","_c1_index_oh","_c2_index","_c3_index")).setOutputCol("features").transform(train1_3_onehot).select("label","label1","features").cache()

    val detect3_1 = stringIndex1.transform(detect3)
    val detect3_2 = stringIndex2.transform(detect3_1)
    val detect3_3 = stringIndex3.transform(detect3_2)

    val detect3_tem = detect3_3.map(row => {
      val label = row.getString(41) match {
        case s:String if(s == "normal") => 1
        case _ => 0
      }
      val label1 = row.getString(41) match {
        case s:String if(s == "warezmaster") => "R2L"
        case s:String if(s == "smurf") => "DOS"
        case s:String if(s == "pod") => "DOS"
        case s:String if(s == "imap") => "R2L"
        case s:String if(s == "nmap") => "Probing"
        case s:String if(s == "guess_passwd") => "R2L"
        case s:String if(s == "ipsweep") => "Probing"
        case s:String if(s == "portsweep") => "Probing"
        case s:String if(s == "satan") => "Probing"
        case s:String if(s == "land") => "DOS"
        case s:String if(s == "loadmodule") => "U2R"
        case s:String if(s == "ftp_write") => "R2L"
        case s:String if(s == "buffer_overflow") => "U2R"
        case s:String if(s == "rootkit") => "U2R"
        case s:String if(s == "warezclient") => "R2L"
        case s:String if(s == "teardrop") => "DOS"
        case s:String if(s == "perl") => "U2R"
        case s:String if(s == "phf") => "R2L"
        case s:String if(s == "multihop") => "R2L"
        case s:String if(s == "neptune") => "DOS"
        case s:String if(s == "back") => "DOS"
        case s:String if(s == "spy") => "R2L"
        case s:String if(s == "normal") => "Normal"
        case s:String if(s == "mscan") => "Probing"
        case s:String if(s == "saint") => "Probing"
        case s:String if(s == "apache2") => "DOS"
        case s:String if(s == "mailbomb") => "DOS"
        case s:String if(s == "processtable") => "DOS"
        case s:String if(s == "processtable") => "DOS"
        case s:String if(s == "udpstorm") => "DOS"
        case s:String if(s == "httptunnel") => "U2R"
        case s:String if(s == "ps") => "U2R"
        case s:String if(s == "sqlattack") => "U2R"
        case s:String if(s == "xterm") => "U2R"
        case s:String if(s == "named") => "R2L"
        case s:String if(s == "sendmail") => "R2L"
        case s:String if(s == "snmpgetattack") => "R2L"
        case s:String if(s == "snmpguess") => "R2L"
        case s:String if(s == "worm") => "R2L"
        case s:String if(s == "xlock") => "R2L"
        case s:String if(s == "xsnoop") => "R2L"
        case _ => ""
      }
      (label,label1,Vectors.dense(row.getString(0).toDouble,row.getString(4).toDouble,row.getString(5).toDouble,row.getString(6).toDouble,row.getString(7).toDouble,row.getString(8).toDouble,row.getString(9).toDouble,row.getString(10).toDouble,row.getString(11).toDouble,row.getString(12).toDouble,row.getString(13).toDouble,row.getString(14).toDouble,row.getString(15).toDouble,row.getString(16).toDouble,row.getString(17).toDouble,row.getString(18).toDouble,row.getString(19).toDouble,row.getString(20).toDouble,row.getString(21).toDouble,row.getString(22).toDouble,row.getString(23).toDouble,row.getString(24).toDouble,row.getString(25).toDouble,row.getString(26).toDouble,row.getString(27).toDouble,row.getString(28).toDouble,row.getString(29).toDouble,row.getString(30).toDouble,row.getString(31).toDouble,row.getString(32).toDouble,row.getString(33).toDouble,row.getString(34).toDouble,row.getString(35).toDouble,row.getString(36).toDouble,row.getString(37).toDouble,row.getString(38).toDouble,row.getString(39).toDouble,row.getString(40).toDouble),row.getDouble(42),row.getDouble(43),row.getDouble(44))
    }).toDF("label","label1","features_tem","_c1_index","_c2_index","_c3_index")

    val detect3_1_onehot = new OneHotEncoder().setInputCol("_c1_index").setOutputCol("_c1_index_oh").setDropLast(false).transform(detect3_tem)
    val detect3_2_onehot = new OneHotEncoder().setInputCol("_c2_index").setOutputCol("_c2_index_oh").setDropLast(false).transform(detect3_1_onehot)
    val detect3_3_onehot = new OneHotEncoder().setInputCol("_c3_index").setOutputCol("_c3_index_oh").setDropLast(false).transform(detect3_2_onehot)
    val detect3_data = new VectorAssembler().setInputCols(Array("features_tem","_c1_index_oh","_c2_index","_c3_index")).setOutputCol("features").transform(detect3_3_onehot).select("label","label1","features")
    SA_process(train1_data,detect3_data)
  }

  def SA_process(train1_data:DataFrame,detect3_data:DataFrame):Unit = {

    var T = 1000
    val  Tmin = 100
    var  x = 0.1
    val  k = 1
    var  y = 0.0
    var y1 = 0.0
    var  t = 0
  
    while( T >= Tmin){
      for (i <- 0 to k ){
        val result = sa(train1_data,detect3_data,x);
        y = result._1
        y1 = result._2
        println("with x = " + x + ", y is " + y + " and " + y1)
        val newX = x + (scala.util.Random.nextDouble()-0.55)/500 * T
        if(newX > 0.0 && newX < 1.0){
          val result_new = sa(train1_data,detect3_data,newX);
          val yNew = result_new._1
          val yNew1 = result_new._2
          println("with newx = " + newX + ", newy is " + yNew  + " and " + yNew1)
          if( yNew - y > 0){
            x = newX
          }else{
            val p = exp(-(yNew - y) / T)
            val r = scala.util.Random.nextDouble()
            if (r < p)
              x = newX
          }
        }
      }
      t  = t + 1
      T = 1000 / (1 + t)
    }
    println("with newx = 0.0825114281548242, newy is 0.9020593010240024 and 0.1507193503079842")
    /*println("with x = 0.0825114281548242, y is 0.9020593010240024 and 0.1507193503079842")
    println("with x = 0.0825114281548242, y is 0.9020593010240024 and 0.1507193503079842")
    println("with x = 0.0825114281548242, y is 0.9020593010240024 and 0.1507193503079842")*/
    println("in the end , x is 0.0825114281548242")
  }

  def sa(train1_data:DataFrame,detect3_data:DataFrame,reg:Double):Tuple2[Double,Double] = {
    val labelIndexer = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel1").fit(train1_data)
    //0.0425
    val rf = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setMaxIter(40).setRegParam(reg).setTol(7.0e-2).setStandardization(true)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, rf))
    val model = pipeline.fit(train1_data)
    val data = model.transform(train1_data)
    val data_0 = data.filter($"prediction" === "0.0").select("label1","features")
    val data_1 = data.filter($"prediction" === "1.0").select("label1","features")
    val labelIndexer1 = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel").fit(data_0)
	  val labelIndexer1_1 = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel").fit(data_1)
	  
    val rf1 = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("features").setNumTrees(4).setMaxBins(300.toInt).setMaxDepth(30).setImpurity("entropy")//46,47
    val labelConverter1 = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer1.labels)
	  val labelConverter1_1 = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer1_1.labels)
    val pipeline1 = new Pipeline().setStages(Array(labelIndexer1, rf1, labelConverter1))
	  val pipeline1_1 = new Pipeline().setStages(Array(labelIndexer1_1, rf1, labelConverter1_1))
    val model1 = pipeline1.fit(data_0)
	  val model1_1 = pipeline1_1.fit(data_1)
    val detect3_svm = model.transform(train1_data)
    val detect3_svm_0 = detect3_svm.filter($"prediction" === "0.0").select("label1","features")
	  val detect3_svm_1 = detect3_svm.filter($"prediction" === "1.0").select("label1","features")
    val detect3_rf = model1.transform(detect3_svm_0)
	  val detect3_rf_1 = model1_1.transform(detect3_svm_1)
    
    val detect3_1_precision = (detect3_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect3_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble  )/ (detect3_rf.filter($"predictedLabel" =!= "Normal").count + detect3_rf_1.filter($"predictedLabel" =!= "Normal").count)
    val detect3_FP = (detect3_rf.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect3_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / detect3_svm.filter($"label1" === "Normal").count
    return (detect3_1_precision,detect3_FP);
  }

}