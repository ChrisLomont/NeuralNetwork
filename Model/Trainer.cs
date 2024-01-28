using System;
using System.Collections.Generic;
using System.Threading;

namespace NeuralNet.Model
{
    // Train a neural net
    public class Trainer
    {
        SimpleNeuralNet neuralNet;
        DataSet dataSet;
        int miniBatchSize = 0;
        float learningRate;
        Action<Result> resultAction;
        Random rand;

        List<int> shuffled;

        private int lockedInterlocked;
        public void Stop()
        {
            Interlocked.Increment(ref lockedInterlocked);
        }

        public struct Result
        {
            public string message;
            public int maxEpochs;
            public int curEpoch;
            public int batchIndex;
            public int numBatches;
            public int trainingSize;
            public int trainingProcessed;
            public float trainingError;
            public int testSize;
            public int testProcessed;
            public float testError;
            public int elapsedMs;
            public string boundsName;
            public float boundsValue;
            public override string ToString()
            {
                return $"Epoch {curEpoch}/{maxEpochs}, batch {batchIndex}/{numBatches}" +
                       $", train & test error ({trainingError:F3},{testError:F3})" +
                       $", Elapsed ms {elapsedMs}, bounds {boundsName}{boundsValue}";
            }
        };

        Result state;

        void Shuffle(Random rand, List<int> items)
        {
            var len = items.Count;
            for (var i = 0; i < len; ++i)
            {
                var j = i + rand.Next(len - i);
                var temp = items[i];
                items[i] = items[j];
                items[j] = temp;
            }
        }
        public void Train(
            // the net to train
            SimpleNeuralNet neuralNet,
            // the data set (training and validation)
            DataSet dataSet,
            // source of randomness
            Random rand,
            // where to put notifications
            Action<Result> resultAction = null,
            // number of epochs to do (complete passes through data)
            int epochs = 100,
            // number to do per mini batch
            int miniBatchSize = 100,
            // learning rate
            float learningRate = 0.1f
        )
        {
            lockedInterlocked = 0;
            this.neuralNet = neuralNet;
            this.dataSet = dataSet;
            this.rand = rand;
            this.state.maxEpochs = epochs;
            this.state.trainingSize = dataSet.TrainingSet.Count;
            this.state.testSize = dataSet.TestSet.Count;
            this.miniBatchSize = miniBatchSize;
            this.learningRate = learningRate;
            this.resultAction = resultAction;
            this.state.message = "";

            state.curEpoch = state.trainingProcessed = state.testProcessed = 0;

            shuffled = new List<int>();
            for (var i = 0; i < state.trainingSize; ++i)
                shuffled.Add(i);

            int startMs = Environment.TickCount, endMs = 0;
            state.numBatches = (state.trainingSize + miniBatchSize - 1) / miniBatchSize;

            for (var epoch = 0; epoch < state.maxEpochs; epoch++)
            {
                Shuffle(rand, shuffled);

                state.curEpoch = epoch + 1;
                state.trainingProcessed = 0;
                state.testProcessed = 0;

                // do training via SGD for one epoch
                var totalTrainingPassed = 0;
                for (var minibatch = 0; minibatch < state.numBatches; ++minibatch)
                {
                    if (lockedInterlocked != 0)
                    {
                        // todo
                        return;
                    }

                    state.batchIndex = minibatch + 1;
                    var batchStart = minibatch * miniBatchSize;
                    var batchSize = Math.Min(miniBatchSize, state.trainingSize - batchStart);
                    totalTrainingPassed += ProcessBatch(
                        i => dataSet.TrainingSet[shuffled[i + batchStart]],
                        batchSize,
                        true
                        );
                    state.trainingProcessed += batchSize;
                    state.trainingError = (float)totalTrainingPassed / state.trainingProcessed;
                    endMs = Environment.TickCount;
                    state.elapsedMs = endMs - startMs;
                    (state.boundsName, state.boundsValue) = neuralNet.Bounds();
                    resultAction(state);
                }

                // check against test data
                var totalTestPassed = ProcessBatch(
                    i => dataSet.TestSet[i],
                    state.testSize,
                    false
                    );
                state.testError = (float)totalTestPassed / state.testSize;
                state.trainingProcessed += this.miniBatchSize;
                endMs = Environment.TickCount;
                state.elapsedMs = endMs - startMs;
                (state.boundsName, state.boundsValue) = neuralNet.Bounds();
                resultAction(state);

            }
        }

        public static int MaxIndex(Vector vector)
        {
            // find max index
            var maxIndex = 0;
            var maxValue = vector[0];
            for (var i = 1; i < vector.Size; ++i)
            {
                if (maxValue < vector[i])
                {
                    maxValue = vector[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        // return true if computed passed compared to desired
        bool Passed(Vector computed, Vector desired)
        {
            return MaxIndex(computed) == MaxIndex(desired);
        }

        // process data, return number correct
        int ProcessBatch(Func<int, DataPoint> getPoint, int batchSize, bool training)
        {

            // run through dataset, total avg error
            var totalPassed = 0;
            if (training)
                neuralNet.ZeroDeltas();
            for (var i = 0; i < batchSize; ++i)
            {
                if (lockedInterlocked != 0)
                {
                    // todo
                    return 0;
                }

                var dataPoint = getPoint(i);
                var computed = neuralNet.FeedForward(dataPoint.input);
                var errorVector = computed - dataPoint.output;
                totalPassed += Passed(computed, dataPoint.output) ? 1 : 0;
                if (training)
                    neuralNet.BackpropagateToDeltas(errorVector);
            }
            if (training)
                neuralNet.ModifyWeights(learningRate, batchSize);
            return totalPassed;
        }
    }
}
