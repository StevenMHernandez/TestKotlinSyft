package com.stevenmhernandez.testkotlinsyft

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import org.openmined.syft.Syft
import org.openmined.syft.domain.SyftConfiguration
import org.openmined.syft.execution.JobStatusSubscriber
import org.openmined.syft.execution.Plan
import org.openmined.syft.networking.datamodels.ClientConfig
import org.openmined.syft.proto.SyftModel
import org.pytorch.IValue
import org.pytorch.Tensor
import java.util.concurrent.ConcurrentHashMap

class MainActivity : AppCompatActivity() {
    @ExperimentalUnsignedTypes
    @ExperimentalStdlibApi
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val localMNISTDataDataSource = LocalMNISTDataDataSource(resources)
        val mnistDataRepository = MNISTDataRepository(localMNISTDataDataSource)

        val userId = "my Id"

        // Optional: Make an http request to your server to get an authentication token
        val authToken = "eyJ0eXAi...."

        // The config defines all the adjustable properties of the syft worker
        // The url entered here cannot define connection protocol like https/wss since the worker allots them by its own
        // `this` supplies the context. It can be an activity context, a service context, or an application context.
        val config = SyftConfiguration.builder(this, "www.mypygrid-url.com").build()

        // Initiate Syft worker to handle all your jobs
        val syftWorker = Syft.getInstance(config, authToken)

        // Create a new Job
        val newJob = syftWorker.newJob("mnist", "1.0.0")

        // Define training procedure for the job
        val jobStatusSubscriber = object : JobStatusSubscriber() {
            override fun onReady(
                    model: SyftModel,
                    plans: ConcurrentHashMap<String, Plan>,
                    clientConfig: ClientConfig
            ) {
                // This function is called when KotlinSyft has downloaded the plans and protocols from PyGrid
                // You are ready to train your model on your data
                // param model stores the model weights given by PyGrid
                // param plans is a HashMap of all the planIDs and their plans.
                // ClientConfig has hyper parameters like batchsize, learning rate, number of steps, etc

                // Plans are accessible by their plan Id used while hosting it on PyGrid.
                // eventually you would be able to use plan name here
                val plan = plans["plan name"]

                repeat(clientConfig.properties.maxUpdates) { step ->

                    // get relevant hyperparams from ClientConfig.planArgs
                    // All the planArgs will be string and it is upon the user to deserialize them into correct type
                    val batchSize = (clientConfig.planArgs["batch_size"]
                            ?: error("batch_size doesn't exist")).toInt()
                    val batchIValue = IValue.from(
                            Tensor.fromBlob(longArrayOf(batchSize.toLong()), longArrayOf(1))
                    )
                    val lr = IValue.from(
                            Tensor.fromBlob(
                                    floatArrayOf(
                                            (clientConfig.planArgs["lr"] ?: error("lr doesn't exist")).toFloat()
                                    ),
                                    longArrayOf(1)
                            )
                    )
                    // your custom implementation to read a databatch from your data
                    val batchData = mnistDataRepository.loadDataBatch(
                            (clientConfig.planArgs["batch_size"] ?: error("batch_size doesn't exist")).toInt()
                    )
                    //get Model weights and return if not set already
                    val modelParams = model.paramArray ?: return
                    val paramIValue = IValue.listFrom(*modelParams)
                    // plan.execute runs a single gradient step and returns the output as PyTorch IValue
                    val output = plan?.execute(
                            batchData.first,
                            batchData.second,
                            batchIValue,
                            lr,paramIValue
                    )?.toTuple()
                    // The output is a tuple with outputs defined by the pysyft plan along with all the model params
                    output?.let { outputResult ->
                        val paramSize = model.stateTensorSize!!
                        // The model params are always appended at the end of the output tuple
                        val beginIndex = outputResult.size - paramSize
                        val updatedParams =
                                outputResult.slice(beginIndex until outputResult.size)
                        // update your model. You can perform any arbitrary computation and checkpoint creation with these model weights
                        model.updateModel(updatedParams)
                        // get the required loss, accuracy, etc values just like you do in Pytorch Android
                        val accuracy = outputResult[0].toTensor().dataAsFloatArray.last()
                    }
                }
                // Once training finishes generate the model diff
                val diff = newJob.createDiff()
                // Report the diff to PyGrid and finish the cycle
                newJob.report(diff)
            }

            fun onRejected() {
                // Implement this function to define what your worker will do when your worker is rejected from the cycle
            }

            override fun onError(throwable: Throwable) {
                // Implement this function to handle error during job execution
            }
        }

        // Start your job
        newJob.start(jobStatusSubscriber)

        // Voila! You are done.
    }
}