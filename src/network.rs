use std::{
    fs::File,
    io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{activations::Activation, matrix::Matrix};

#[derive(Serialize, Deserialize)]
struct SaveData {
    inputs: usize,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
    learning_rate: f64,
}

/**
 * The Neural Network class
 */
pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl Network<'_> {
    /**

    The Neural Network constructor.

    # Arguments

    * `layers`: The sizes of the layers
    * `learning_rate`: The training learning rate
    * `activation`: The Activation struct containing the the activation function and his derivated

    # Examples

    ```
    let mut nn = k_ai::network::Network::new(
        &[2, 4, 1],
        0.1,
        k_ai::activations::SIGMOID,
    );
    ```
    */
    pub fn new<'a>(
        layers: &[usize],
        learning_rate: f64,
        activation: Activation<'a>,
    ) -> Network<'a> {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers: Vec::from(layers),
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to touch save file");

        file.write_all(
			json!({
                "inputs": self.layers[0],
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
                "learning_rate": self.learning_rate,
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
    }

    pub fn load<'a>(file: String, activation: Activation<'a>) -> Network<'a> {
        let mut file = File::open(file).expect("Unable to open save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

        let mut weights = vec![];
        let mut biases = vec![];
        let mut layers: Vec<usize> = vec![save_data.inputs];

        for i in 0..save_data.weights.len() {
            layers.push(save_data.weights[i].len());

            weights.push(Matrix::from(save_data.weights[i].clone()));
            biases.push(Matrix::from(save_data.biases[i].clone()));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate: save_data.learning_rate,
        }
    }

    /**

    Result the output of a given input array.

    # Arguments

    * `inputs`: The inputs array to feed forward

    # Examples

    ```
    let mut nn = k_ai::network::Network::new(
        &[2, 4, 1],
        0.1,
        k_ai::activations::SIGMOID,
    );

    nn.feed_forward(&[0.0, 1.0]);
    ```
    */
    pub fn feed_forward(&mut self, inputs: &[f64]) -> Result<Vec<f64>, &str> {
        if inputs.len() != self.layers[0] {
            return Err("Invalid number of inputs");
        }

        let mut current = Matrix::from(vec![Vec::from(inputs)]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.func);
            self.data.push(current.clone());
        }

        Ok(current.data[0].to_owned())
    }

    /**

    Back propagate and correct from the actual output and and the targeted output.

    # Arguments

    * `outputs`: The actual output
    * `targets`: The wanted output

    # Examples

    ```
    let mut nn = k_ai::network::Network::new(
        &[2, 4, 1],
        0.1,
        k_ai::activations::SIGMOID,
    );

    nn.back_propagate(vec![0.035445333], vec![1.0]);
    ```
    */
    fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> Result<(), &str> {
        if targets.len() != self.layers[self.layers.len() - 1] {
            return Err("Invalid number of targets");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = parsed.map(self.activation.dfunc);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .dot_multiply(&errors)
                .map(&|x| x * self.learning_rate);

            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.dfunc);
        }

        Ok(())
    }

    /**

    Train the Neural Network from an array of DatasetValue.

    # Arguments

    * `dataset`: The dataset to train from
    * `iter`: The number of how many time to train the Neural Network

    # Examples

    ```
    use k_ai::{
        activations::SIGMOID,
        network::{DatasetValue, Network},
    };

    let mut nn = Network::new(
        &[2, 4, 1],
        0.1,
        SIGMOID,
    );

    nn.train(&[
        DatasetValue {
            inputs: &[0.0, 0.0],
            targets: &[0.0],
        },
        DatasetValue {
            inputs: &[0.0, 1.0],
            targets: &[1.0],
        },
        DatasetValue {
            inputs: &[1.0, 0.0],
            targets: &[1.0],
        },
        DatasetValue {
            inputs: &[1.0, 1.0],
            targets: &[0.0],
        },
    ], 1_000);
    ```
    */
    pub fn train(&mut self, dataset: &[DatasetValue], iter: u16) {
        for i in 1..=iter {
            if iter < 100 || i % (iter / 100) == 0 {
                println!("Iteration {} of {}", i, iter);
            }

            for j in 0..dataset.len() {
                let outputs = self.feed_forward(dataset[j].inputs).unwrap();
                self.back_propagate(outputs, Vec::from(dataset[j].targets))
                    .unwrap();
            }
        }
    }
}

#[derive(Clone)]
pub struct DatasetValue<'a> {
    pub inputs: &'a [f64],
    pub targets: &'a [f64],
}
