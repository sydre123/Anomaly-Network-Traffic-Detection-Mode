package com.dsj.spark.testSpark
import java.util.Date
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
//import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.VectorAssembler
object Model {
  val master = "local";
  val appName= "model";
  val spark = SparkSession.builder().master(master).appName(appName).getOrCreate();
  val sc = spark.sparkContext
  import spark.implicits._
  def main(args:Array[String]) :Unit = {
    //最终模型
    //  屏蔽日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    if(args.length < 5){
      println("you should give four parameters ,with the path of  train1,detect1,detect2 !")
      return;
    }
//    val train1 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter", ",").load("file:///home/hadoop/test/source/kdd99/train3/data.csv")
//    val detect1 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter",",").load("file:///home/hadoop/test/source/kdd99/detect1/data.csv")
//    val detect2 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter",",").load("file:///home/hadoop/test/source/kdd99/detect2/data.csv")
    val train1 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter", ",").load(args(0))
    val detect1 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter",",").load(args(1))
    val detect2 = spark.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "false").option("delimiter",",").load(args(2))

    val stringIndex1 =  new StringIndexer().setInputCol("_c1").setOutputCol("_c1_index").fit(train1.select($"_c1").union(detect1.select($"_c1")).union(detect2.select($"_c1")))
    val stringIndex2 =  new StringIndexer().setInputCol("_c2").setOutputCol("_c2_index").fit(train1.select($"_c2").union(detect2.select($"_c2")).union(detect1.select($"_c2")))
    val stringIndex3 =  new StringIndexer().setInputCol("_c3").setOutputCol("_c3_index").fit(train1.select($"_c3").union(detect1.select($"_c3")).union(detect2.select($"_c3")))

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

    train1_data.show(false)
    /*-------------------svm--------------------------------*/
    val labelIndexer = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel1").fit(train1_data)
    //0.0425----args(4).toDouble
    val rf = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setMaxIter(40).setRegParam(args(3).toDouble).setTol(7.0e-2).setStandardization(true)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, rf))
    var start_time =new Date().getTime
    val model = pipeline.fit(train1_data)
    val data = model.transform(train1_data)
    val data_0 = data.filter($"prediction" === "0.0").select("label1","features")
    val data_1 = data.filter($"prediction" === "1.0").select("label1","features")
    /*-------------------rf-----------------------------------*/
    val labelIndexer1 = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel").fit(data_0)
	  val labelIndexer1_1 = new StringIndexer().setInputCol("label1").setOutputCol("indexedLabel").fit(data_1)
    //300----args(5).toInt
    val rf1 = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("features").setNumTrees(4).setMaxBins(args(4).toInt).setMaxDepth(30).setImpurity("entropy")//46,47
    val labelConverter1 = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer1.labels)
	  val labelConverter1_1 = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer1_1.labels)
    val pipeline1 = new Pipeline().setStages(Array(labelIndexer1, rf1, labelConverter1))
	  val pipeline1_1 = new Pipeline().setStages(Array(labelIndexer1_1, rf1, labelConverter1_1))
    val model1 = pipeline1.fit(data_0)
	  val model1_1 = pipeline1_1.fit(data_1)
    val train1_svm = model.transform(train1_data)
    val train1_svm_0 = train1_svm.filter($"prediction" === "0.0").select("label1","features")
	  val train1_svm_1 = train1_svm.filter($"prediction" === "1.0").select("label1","features")
    val train1_rf = model1.transform(train1_svm_0)
	  val train1_rf_1 = model1_1.transform(train1_svm_1)
    var end_time =new Date().getTime
    
     //accuracy,recall,Precision
    //val train1_acc = (train1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + train1_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + train1_svm.filter($"label1" === "Normal").filter($"prediction" === 1.0).count )/ (train1_rf.count + train1_svm.filter($"prediction" === 1.0).count )
    //val train1_1_recall = (train1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + train1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / train1_svm.filter($"label1" =!= "Normal").count
    val train1_1_precision = (train1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + train1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / (train1_rf.filter($"predictedLabel" =!= "Normal").count + train1_rf_1.filter($"predictedLabel" =!= "Normal").count)
    //误报率FP,漏报率FN
    val train1_FP = (train1_rf.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + train1_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / train1_svm.filter($"label1" === "Normal").count
    //val train1_FN = (train1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble + train1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble) / train1_svm.filter($"label1" =!= "Normal").count

    //各类别的准确率和误报率
    //val train1_Normal_pre = (train1_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + train1_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble )/ (train1_rf.filter($"predictedLabel" === "Normal").count + train1_rf_1.filter($"predictedLabel" === "Normal").count)
    //val train1_Normal_fp = (train1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count + train1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count) / ( train1_svm.filter($"label1" =!= "Normal").count )
    val train1_Probing_pre =  (train1_rf.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble + train1_rf_1.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble  ) / (train1_rf.filter($"predictedLabel" === "Probing").count + train1_rf_1.filter($"predictedLabel" === "Probing").count)
    //val train1_Probing_fp =  (train1_rf.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble + train1_rf_1.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble ) / ( train1_svm.filter($"label1" =!= "Probing").count )
    val train1_DOS_pre =  (train1_rf.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble + train1_rf_1.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble )/ (train1_rf.filter($"predictedLabel" === "DOS").count + train1_rf_1.filter($"predictedLabel" === "DOS").count)
    //val train1_DOS_fp = (train1_rf.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble + train1_rf_1.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble) / ( train1_svm.filter($"label1" =!= "DOS").count )
    val train1_R2L_pre = (train1_rf.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble + train1_rf_1.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble )/ (train1_rf.filter($"predictedLabel" === "R2L").count + train1_rf_1.filter($"predictedLabel" === "R2L").count)
    //val train1_R2L_fp = (train1_rf.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble + train1_rf_1.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble)/ ( train1_svm.filter($"label1" =!= "R2L").count )
    val train1_U2R_pre = (train1_rf.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble + train1_rf_1.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble )/ (train1_rf.filter($"predictedLabel" === "U2R").count + train1_rf_1.filter($"predictedLabel" === "U2R").count)
    //val train1_U2R_fp =  (train1_rf.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble + train1_rf_1.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble)/ ( train1_svm.filter($"label1" =!= "U2R").count )
    
    println("----------------------- now is train1 --------------------")
    println("-----------------------------整体效果-----------------------")
    //println("The accuracy of train1 is : "+train1_acc)
    //println("the recall of train1 is : " + train1_1_recall)
    println("the precision of train1 is : " + train1_1_precision)
    println("the False Positive  rate of train1 is : " + train1_FP)
    //println("the False Negative rate of train1 is : " + train1_FN)
    println("to use  : " + (end_time-start_time).toString + "ms to train the data")
    println("-----------------------------各类别-----------------------")
    //println("the Precision of normal is : "+ train1_Normal_pre )
    //println("the False Positive rate of normal is : "+ train1_Normal_fp )
    println("the Precision of Probing is : "+ train1_Probing_pre )
    //println("the False Positive rate of Probing is : "+ train1_Probing_fp )
    println("the Precision of DOS is : "+ train1_DOS_pre )
    //println("the False Positive rate of DOS is : "+ train1_DOS_fp )
    println("the Precision of R2L is : "+ train1_R2L_pre )
    //println("the False Positive rate of R2L is : "+ train1_R2L_fp )
    println("the Precision of U2R is : "+ train1_U2R_pre )
    //println("the False Positive rate of U2R is : "+ train1_U2R_fp )
    
    /*------------------------------ detect1 --------------------------------- */
    val detect1_1 = new StringIndexer().setInputCol("_c1").setOutputCol("_c1_index").fit(detect1).transform(detect1)
    val detect1_2 = new StringIndexer().setInputCol("_c2").setOutputCol("_c2_index").fit(detect1_1).transform(detect1_1)
    val detect1_3 = new StringIndexer().setInputCol("_c3").setOutputCol("_c3_index").fit(detect1_2).transform(detect1_2)

    val detect1_tem = detect1_3.map(row => {
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

    val detect1_1_onehot = new OneHotEncoder().setInputCol("_c1_index").setOutputCol("_c1_index_oh").setDropLast(false).transform(detect1_tem)
    val detect1_2_onehot = new OneHotEncoder().setInputCol("_c2_index").setOutputCol("_c2_index_oh").setDropLast(false).transform(detect1_1_onehot)
    val detect1_3_onehot = new OneHotEncoder().setInputCol("_c3_index").setOutputCol("_c3_index_oh").setDropLast(false).transform(detect1_2_onehot)
    val detect1_data = new VectorAssembler().setInputCols(Array("features_tem","_c1_index_oh","_c2_index","_c3_index")).setOutputCol("features").transform(detect1_3_onehot).select("label","label1","features").cache()
    start_time =new Date().getTime
    val detect1_svm = model.transform(detect1_data)
    val detect1_svm_0 = detect1_svm.filter($"prediction" === "0.0").select("label1","features")
	  val detect1_svm_1 = detect1_svm.filter($"prediction" === "1.0").select("label1","features")
    val detect1_rf = model1.transform(detect1_svm_0)
	  val detect1_rf_1 = model1_1.transform(detect1_svm_1)
    end_time =new Date().getTime
    
     //accuracy,racall,precision
    //val detect1_acc = (detect1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect1_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect1_svm.filter($"label1" === "Normal").filter($"prediction" === 1.0).count )/ (detect1_rf.count + detect1_svm.filter($"prediction" === 1.0).count )
    //val detect1_1_recall = (detect1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / detect1_svm.filter($"label1" =!= "Normal").count
    val detect1_1_precision = (detect1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble  )/ (detect1_rf.filter($"predictedLabel" =!= "Normal").count + detect1_rf_1.filter($"predictedLabel" =!= "Normal").count)
    //误报率FP,漏报率FN
    val detect1_FP = (detect1_rf.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect1_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / detect1_svm.filter($"label1" === "Normal").count
    //val detect1_FN = (detect1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble) / detect1_svm.filter($"label1" =!= "Normal").count

    //给类别的准确率和误报率
    //val detect1_Normal_pre = (detect1_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect1_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble )/ (detect1_rf.filter($"predictedLabel" === "Normal").count + detect1_rf_1.filter($"predictedLabel" === "Normal").count)
    //val detect1_Normal_fp = (detect1_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count + detect1_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count) / ( detect1_svm.filter($"label1" =!= "Normal").count )
    val detect1_Probing_pre =  (detect1_rf.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble + detect1_rf_1.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble  )/ (detect1_rf.filter($"predictedLabel" === "Probing").count + detect1_rf_1.filter($"predictedLabel" === "Probing").count )
    //val detect1_Probing_fp =  (detect1_rf.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble + detect1_rf_1.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble )/ ( detect1_svm.filter($"label1" =!= "Probing").count )
    val detect1_DOS_pre =  (detect1_rf.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble + detect1_rf_1.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble )/ (detect1_rf.filter($"predictedLabel" === "DOS").count + detect1_rf_1.filter($"predictedLabel" === "DOS").count )
    //val detect1_DOS_fp = (detect1_rf.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble + detect1_rf_1.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble)/ ( detect1_svm.filter($"label1" =!= "DOS").count )
    val detect1_R2L_pre = (detect1_rf.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble + detect1_rf_1.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble )/ (detect1_rf.filter($"predictedLabel" === "R2L").count + detect1_rf_1.filter($"predictedLabel" === "R2L").count )
    //val detect1_R2L_fp = (detect1_rf.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble + detect1_rf_1.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble)/ ( detect1_svm.filter($"label1" =!= "R2L").count )
    val detect1_U2R_pre = (detect1_rf.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble + detect1_rf_1.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble )/ (detect1_rf.filter($"predictedLabel" === "U2R").count + detect1_rf_1.filter($"predictedLabel" === "U2R").count )
    //val detect1_U2R_fp =  (detect1_rf.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble + detect1_rf_1.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble)/ ( detect1_svm.filter($"label1" =!= "U2R").count )

    println("----------------------- now is detect1 --------------------")
    println("to use : " + (end_time-start_time).toString + "ms to test the data")
    println("-----------------------------整体效果-----------------------")
    //println("The accuracy of detect1 is : "+detect1_acc)//0.891729950319375414
    //println("the recall of detect1 is : " + detect1_1_recall)//0.891781344969999231
    println("the precision of detect1 is : " + detect1_1_precision)
    println("the False Positive  rate of detect1 is : " + detect1_FP)
    //println("the False Negative rate of detect1 is : " +detect1_FN )//0.018218655030000840
 
    println("-----------------------------各类别-----------------------")
    //println("the Precision of normal is : "+ detect1_Normal_pre )
    //println("the False Positive rate of normal is : "+ detect1_Normal_fp )
    println("the Precision of Probing is : "+ detect1_Probing_pre )
    //println("the False Positive rate of Probing is : "+ detect1_Probing_fp )
    println("the Precision of DOS is : "+ detect1_DOS_pre )
    //println("the False Positive rate of DOS is : "+ detect1_DOS_fp )
    println("the Precision of R2L is : "+ detect1_R2L_pre )
    //println("the False Positive rate of R2L is : "+ detect1_R2L_fp )
    println("the Precision of U2R is : "+ detect1_U2R_pre )
    //println("the False Positive rate of U2R is : "+ detect1_U2R_fp )

    /*-------------------------------- detect2 -----------------------*/
    val detect2_1 = new StringIndexer().setInputCol("_c1").setOutputCol("_c1_index").fit(detect2).transform(detect2)
    val detect2_2 = new StringIndexer().setInputCol("_c2").setOutputCol("_c2_index").fit(detect2_1).transform(detect2_1)
    val detect2_3 = new StringIndexer().setInputCol("_c3").setOutputCol("_c3_index").fit(detect2_2).transform(detect2_2)
    val detect2_tem = detect2_3.map(row => {
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

    val detect2_1_onehot = new OneHotEncoder().setInputCol("_c1_index").setOutputCol("_c1_index_oh").setDropLast(false).transform(detect2_tem)
    val detect2_2_onehot = new OneHotEncoder().setInputCol("_c2_index").setOutputCol("_c2_index_oh").setDropLast(false).transform(detect2_1_onehot)
    val detect2_3_onehot = new OneHotEncoder().setInputCol("_c3_index").setOutputCol("_c3_index_oh").setDropLast(false).transform(detect2_2_onehot)
    val detect2_data = new VectorAssembler().setInputCols(Array("features_tem","_c1_index_oh","_c2_index","_c3_index")).setOutputCol("features").transform(detect2_3_onehot).select("label","label1","features")

    start_time =new Date().getTime
    val detect2_svm = model.transform(detect2_data)
    val detect2_svm_0 = detect2_svm.filter($"prediction" === 0.0).select("label1","features")
    val detect2_svm_1 = detect2_svm.filter($"prediction" === 1.0).select("label1","features")
    val detect2_rf = model1.transform(detect2_svm_0)
    val detect2_rf_1 = model1_1.transform(detect2_svm_1)
    //val detect2_num = 1
    end_time =new Date().getTime    
    
    //accuracy,racall,precision
    //val detect2_acc = (detect2_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect2_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect2_svm.filter($"label1" === "Normal").filter($"prediction" === 1.0).count )/ (detect2_rf.count + detect2_svm.filter($"prediction" === 1.0).count )
    //val detect2_1_recall = (detect2_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect2_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / detect2_svm.filter($"label1" =!= "Normal").count
    val detect2_1_precision = (detect2_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect2_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble  )/ (detect2_rf.filter($"predictedLabel" =!= "Normal").count + detect2_rf_1.filter($"predictedLabel" =!= "Normal").count)
    //误报率FP,漏报率FN
    val detect2_FP = (detect2_rf.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble + detect2_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" =!= "Normal").count.toDouble) / detect2_svm.filter($"label1" === "Normal").count
    //val detect2_FN = (detect2_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect2_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count.toDouble) / detect2_svm.filter($"label1" =!= "Normal").count

    //给类别的准确率和误报率
    //val detect2_Normal_pre = (detect2_rf.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble + detect2_rf_1.filter($"label1" === "Normal").filter($"predictedLabel" === "Normal").count.toDouble )/ (detect2_rf.filter($"predictedLabel" === "Normal").count + detect2_rf_1.filter($"predictedLabel" === "Normal").count)
    //val detect2_Normal_fp = (detect2_rf.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count + detect2_rf_1.filter($"label1" =!= "Normal").filter($"predictedLabel" === "Normal").count) / ( detect2_svm.filter($"label1" =!= "Normal").count )
    val detect2_Probing_pre =  (detect2_rf.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble + detect2_rf_1.filter($"label1" === "Probing").filter($"predictedLabel" === "Probing").count.toDouble  )/ (detect2_rf.filter($"predictedLabel" === "Probing").count + detect2_rf_1.filter($"predictedLabel" === "Probing").count )
    //val detect2_Probing_fp =  (detect2_rf.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble + detect2_rf_1.filter($"label1" =!= "Probing").filter($"predictedLabel" === "Probing").count.toDouble )/ ( detect2_svm.filter($"label1" =!= "Probing").count )
    val detect2_DOS_pre =  (detect2_rf.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble + detect2_rf_1.filter($"label1" === "DOS").filter($"predictedLabel" === "DOS").count.toDouble )/ (detect2_rf.filter($"predictedLabel" === "DOS").count + detect2_rf_1.filter($"predictedLabel" === "DOS").count )
    //val detect2_DOS_fp = (detect2_rf.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble + detect2_rf_1.filter($"label1" =!= "DOS").filter($"predictedLabel" === "DOS").count.toDouble)/ ( detect2_svm.filter($"label1" =!= "DOS").count )
    val detect2_R2L_pre = (detect2_rf.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble + detect2_rf_1.filter($"label1" === "R2L").filter($"predictedLabel" === "R2L").count.toDouble )/ (detect2_rf.filter($"predictedLabel" === "R2L").count + detect2_rf_1.filter($"predictedLabel" === "R2L").count )
    //val detect2_R2L_fp = (detect2_rf.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble + detect2_rf_1.filter($"label1" =!= "R2L").filter($"predictedLabel" === "R2L").count.toDouble)/ ( detect2_svm.filter($"label1" =!= "R2L").count )
    val detect2_U2R_pre = (detect2_rf.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble + detect2_rf_1.filter($"label1" === "U2R").filter($"predictedLabel" === "U2R").count.toDouble )/ (detect2_rf.filter($"predictedLabel" === "U2R").count + detect2_rf_1.filter($"predictedLabel" === "U2R").count )
    //val detect2_U2R_fp =  (detect2_rf.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble + detect2_rf_1.filter($"label1" =!= "U2R").filter($"predictedLabel" === "U2R").count.toDouble)/ ( detect2_svm.filter($"label1" =!= "U2R").count )

    println("----------------------- now is detect2 --------------------") 
    println("to use : " + (end_time-start_time).toString + "ms to test the data")
    println("-----------------------------整体效果-----------------------")
    //println("The accuracy of detect2 is : "+detect2_acc)//0.788607594936708826
    //println("the recall of detect2 is : " + detect2_1_recall)//0.8875149515363992544
    println("the precision of detect2 is : " + detect2_1_precision)
    println("the False Positive  rate of detect2 is : " + detect2_FP)
    //println("the False Negative rate of detect2 is : " + detect2_FN)//0.1524850484636007
    
    println("-----------------------------各类别-----------------------")
    //println("the Precision of normal is : "+ detect2_Normal_pre )
    //println("the False Positive rate of normal is : "+ detect2_Normal_fp )
    println("the Precision of Probing is : "+ detect2_Probing_pre )
    //println("the False Positive rate of Probing is : "+ detect2_Probing_fp )
    println("the Precision of DOS is : "+ detect2_DOS_pre )
    //println("the False Positive rate of DOS is : "+ detect2_DOS_fp )
    println("the Precision of R2L is : "+ detect2_R2L_pre )
    //println("the False Positive rate of R2L is : "+ detect2_R2L_fp )
    println("the Precision of U2R is : "+ detect2_U2R_pre )
    //println("the False Positive rate of U2R is : "+ detect2_U2R_fp )

  }
  
}