use super::{activations::Activation, matrix::Matrix};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl Network<'_> {
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

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> Result<(), &str> {
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

    pub fn train(&mut self, dataset: &[DataSetValue], epochs: u16) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }

            for j in 0..dataset.len() {
                let outputs = self.feed_forward(dataset[j].inputs).unwrap();
                self.back_propagate(outputs, Vec::from(dataset[j].targets)).unwrap();
            }
        }
    }
}

#[derive(Clone)]
pub struct DataSetValue<'a> {
    pub inputs: &'a [f64],
    pub targets: &'a [f64],
}
