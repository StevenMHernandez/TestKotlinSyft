package com.stevenmhernandez.testkotlinsyft

import com.stevenmhernandez.testkotlinsyft.LocalMNISTDataDataSource
import org.pytorch.IValue
import org.pytorch.Tensor

class MNISTDataRepository constructor(
    private val localMNISTDataDataSource: LocalMNISTDataDataSource
) {
    fun loadDataBatch(batchSize: Int): Pair<IValue, IValue> {
        val data = localMNISTDataDataSource.loadDataBatch(batchSize)
        val tensorsX = IValue.from(Tensor.fromBlob(data.first.flattenedArray, data.first.shape))

        val tensorsY = IValue.from(Tensor.fromBlob(data.second.flattenedArray, data.second.shape))
        return Pair(tensorsX, tensorsY)
    }
}