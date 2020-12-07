package com.dsj.spark.testSpark
import com.google.common.eventbus.Subscribe
import org.spark_project.guava.eventbus.Subscribe
object stream {
  def main(args: Array[String]): Unit = {

//sc.stop() 
import scala.util.parsing.json.JSON
import java.util.Properties
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, ConsumerStrategies, HasOffsetRanges, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}
//   import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.streaming.kafka010._
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
    var conf=new SparkConf().setMaster("local[*]").setAppName("SparkStreamKafla").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    val sc = new SparkContext(conf)
    val sparkSession: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import sparkSession.implicits._
    val svm = PipelineModel.load("file:///home/hadoop/app/spark/svm")
    val rf = PipelineModel.load("file:///home/hadoop/app/spark/rf")
    var ssc=new StreamingContext(sc,Seconds(5));
    //创建topic
    //var topic=Map{"test" -> 1}

    var topic=Array("test");
    //指定zookeeper
    //创建消费者组
    var group="test-consumer-group"
    //消费者配置
    val kafkaParam = Map(
      "bootstrap.servers" -> "127.0.0.1:9092",//用于初始化链接到集群的地址
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      //            "key.deserializer" -> classOf[String],
      //            "value.deserializer" -> classOf[String],
      //用于标识这个消费者属于哪个消费团体
      "group.id" -> group,
      //如果没有初始化偏移量或者当前的偏移量不存在任何服务器上，可以使用这个配置属性
      //可以使用这个配置，latest自动重置偏移量为最新的偏移量
      "auto.offset.reset" -> "latest",
      //如果是true，则这个消费者的偏移量会在后台自动提交
      "enable.auto.commit" -> (false: java.lang.Boolean)
    );
    //创建DStream，返回接收到的输入数据
    var stream=KafkaUtils.createDirectStream[String,String](ssc, LocationStrategies.PreferConsistent,ConsumerStrategies.Subscribe[String,String](topic,kafkaParam))

    val properties = new Properties()
    properties.setProperty("user","root")
    properties.setProperty("password","")
    //每一个stream都是一个ConsumerRecord
    stream.map(s =>(s.value())).foreachRDD(rdd => {
      val df = rdd.map(ln =>{
        val row = ln.split(",")
        val _c1_index = row(1) match {
          case s:String if(s == "tcp") => Vectors.sparse(3,Seq((0,1.0)))
          case s:String if(s == "udp") => Vectors.sparse(3,Seq((1,1.0)))
          case s:String if(s == "icmp") => Vectors.sparse(3,Seq((2,1.0)))
          case _ => Vectors.sparse(3,Seq((0,1.0)))
        }

        val _c2_index = row(2) match {
          case s:String if(s=="telnet")=>11.0
          case s:String if(s=="ftp")=>8.0
          case s:String if(s=="auth")=>14.0
          case s:String if(s=="iso_tsap")=>24.0
          case s:String if(s=="systat")=>22.0
          case s:String if(s=="name")=>39.0
          case s:String if(s=="sql_net")=>48.0
          case s:String if(s=="ntp_u")=>12.0
          case s:String if(s=="X11")=>20.0
          case s:String if(s=="pop_3")=>13.0
          case s:String if(s=="discard")=>51.0
          case s:String if(s=="tftp_u")=>65.0
          case s:String if(s=="Z39_50")=>36.0
          case s:String if(s=="daytime")=>50.0
          case s:String if(s=="domain_u")=>4.0
          case s:String if(s=="login")=>33.0
          case s:String if(s=="smtp")=>3.0
          case s:String if(s=="mtp")=>28.0
          case s:String if(s=="domain")=>25.0
          case s:String if(s=="http")=>0.0
          case s:String if(s=="link")=>27.0
          case s:String if(s=="courier")=>49.0
          case s:String if(s=="pop_2")=>58.0
          case s:String if(s=="other")=>5.0
          case s:String if(s=="efs")=>26.0
          case s:String if(s=="nnsp")=>32.0
          case s:String if(s=="IRC")=>16.0
          case s:String if(s=="pm_dump")=>63.0
          case s:String if(s=="private")=>1.0
          case s:String if(s=="urh_i")=>19.0
          case s:String if(s=="ftp_data")=>6.0
          case s:String if(s=="whois")=>57.0
          case s:String if(s=="nntp")=>41.0
          case s:String if(s=="netbios_ns")=>62.0
          case s:String if(s=="klogin")=>52.0
          case s:String if(s=="shell")=>31.0
          case s:String if(s=="red_i")=>64.0
          case s:String if(s=="tim_i")=>55.0
          case s:String if(s=="uucp_path")=>61.0
          case s:String if(s=="eco_i")=>7.0
          case s:String if(s=="ctf")=>60.0
          case s:String if(s=="vmnet")=>45.0
          case s:String if(s=="supdup")=>35.0
          case s:String if(s=="finger")=>10.0
          case s:String if(s=="printer")=>38.0
          case s:String if(s=="urp_i")=>9.0
          case s:String if(s=="ecr_i")=>2.0
          case s:String if(s=="time")=>15.0
          case s:String if(s=="netbios_ss")=>30.0
          case s:String if(s=="hostnames")=>37.0
          case s:String if(s=="csnet_ns")=>40.0
          case s:String if(s=="sunrpc")=>18.0
          case s:String if(s=="echo")=>44.0
          case s:String if(s=="http_443")=>56.0
          case s:String if(s=="netstat")=>21.0
          case s:String if(s=="remote_jo")=>23.0
          case s:String if(s=="gopher")=>43.0
          case s:String if(s=="imap4")=>17.0
          case s:String if(s=="uucp")=>29.0
          case s:String if(s=="ssh")=>34.0
          case s:String if(s=="rje")=>47.0
          case s:String if(s=="bgp")=>42.0
          case _ => -1
        }

        val _c3_index = row(3) match {
          case s:String if(s == "RSTOS0") =>  9.0
          case s:String if(s == "S3") =>     7.0
          case s:String if(s == "SF") =>     0.0
          case s:String if(s == "S0") =>     2.0
          case s:String if(s == "OTH") =>10.0
          case s:String if(s == "REJ") =>1.0
          case s:String if(s == "RSTO") =>4.0
          case s:String if(s == "RSTR") =>3.0
          case s:String if(s == "SH") =>      6.0
          case s:String if(s == "S2") =>     8.0
          case s:String if(s == "S1") =>     5.0
          case _ => -1
        }

        (ln,1,"Normal",Vectors.dense(row(0).toDouble,row(4).toDouble,row(5).toDouble,row(6).toDouble,row(7).toDouble,row(8).toDouble,row(9).toDouble,row(10).toDouble,row(11).toDouble,row(12).toDouble,row(13).toDouble,row(14).toDouble,row(15).toDouble,row(16).toDouble,row(17).toDouble,row(18).toDouble,row(19).toDouble,row(20).toDouble,row(21).toDouble,row(22).toDouble,row(23).toDouble,row(24).toDouble,row(25).toDouble,row(26).toDouble,row(27).toDouble,row(28).toDouble,row(29).toDouble,row(30).toDouble,row(31).toDouble,row(32).toDouble,row(33).toDouble,row(34).toDouble,row(35).toDouble,row(36).toDouble,row(37).toDouble,row(38).toDouble,row(39).toDouble,row(40).toDouble),_c1_index,_c2_index,_c3_index)
      }).toDF("ln","label","label1","features_tem","_c1_index_oh","_c2_index","_c3_index")
      //      df.show
      df.show
      //      val train1_1_onehot = new OneHotEncoder().setInputCol("_c1_index").setOutputCol("_c1_index_oh").setDropLast(false).transform(df)
      //      val train1_2_onehot = new OneHotEncoder().setInputCol("_c2_index").setOutputCol("_c2_index_oh").setDropLast(false).transform(train1_1_onehot)
      //      val train1_3_onehot = new OneHotEncoder().setInputCol("_c3_index").setOutputCol("_c3_index_oh").setDropLast(false).transform(train1_2_onehot)
      val train1_data = new VectorAssembler().setInputCols(Array("features_tem","_c1_index_oh","_c2_index","_c3_index")).setOutputCol("features").transform(df).select("label","label1","features","ln").cache()
      //      train1_data.show
      val svm_data = svm.transform(train1_data)
      val r1 = svm_data.filter($"prediction" === "1.0").groupBy("prediction").count.map(row =>{
        val label = "Normal"
        val quantity = row.getLong(1)
        (label,quantity,new java.sql.Timestamp(System.currentTimeMillis()))
      }).toDF("type","quantity","the_time")
      val rf_data = rf.transform(svm_data.filter($"prediction" === "0.0").select("label1","features","ln"))
      val r2 = rf_data.groupBy("predictedLabel").count.map(row =>{
        val label = row.getString(0)
        val quantity = row.getLong(1)
        (label,quantity,new java.sql.Timestamp(System.currentTimeMillis()))
      }).toDF("type","quantity","the_time")

      val r3 = svm_data.filter($"prediction" === "1.0").select("ln").map(row => {
        val label = "Normal"
        val ln = row.getString(0)
        (label,ln,new java.sql.Timestamp(System.currentTimeMillis()))
      }).toDF("type","detail","the_time")
      val r4 = rf_data.select("ln","predictedLabel").map(row => {
        val label = row.getString(1)
        val ln = row.getString(0)
        (label,ln,new java.sql.Timestamp(System.currentTimeMillis()))
      }).toDF("type","detail","the_time")
      //      svm_data.show
      r1.write.mode("append").jdbc("jdbc:mysql://192.168.79.1:3306/network?useUnicode=true&characterEncoding=utf8","flow_quantity",properties)
      r2.write.mode("append").jdbc("jdbc:mysql://192.168.79.1:3306/network?useUnicode=true&characterEncoding=utf8","flow_quantity",properties)
      r3.write.mode("append").jdbc("jdbc:mysql://192.168.79.1:3306/network?useUnicode=true&characterEncoding=utf8","flow_time",properties)
      r4.write.mode("append").jdbc("jdbc:mysql://192.168.79.1:3306/network?useUnicode=true&characterEncoding=utf8","flow_time",properties)

      train1_data.unpersist()
    })
    //    stream.map(s =>(s.key(),s.value())).print();
    //    stream.print()


    ssc.start();
    ssc.awaitTermination();



  }
  
}