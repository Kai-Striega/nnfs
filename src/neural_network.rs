use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

struct NeuralNetwork {
    n_layers: usize,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}


impl NeuralNetwork {
    pub fn from_sizes(sizes: Vec<usize>) -> Self {
        let n_layers = sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        let layer_size_iter = sizes
            .iter()
            .zip(sizes.iter().skip(1));
        let rng = Normal::new(0.0, 1.0).expect("could not create rng");

        for (in_nodes, out_nodes) in layer_size_iter {
            let weight = Array2::random((*in_nodes, *out_nodes), rng);
            weights.push(weight);
            let bias = Array1::random(*out_nodes, rng);
            biases.push(bias);
        }

        Self { n_layers, weights, biases }
    }

    fn forward_propagate(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut output = x.clone();
        let layer_iter = self.weights.iter().zip(self.biases.iter());
        for (weight, bias) in layer_iter {
            let intermediate_output: Array1<f64> = weight.t().dot(&output) + bias;
            output = intermediate_output.mapv(sigmoid)
        }
        output
    }
}
