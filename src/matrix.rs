use rand::random;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut res: Self = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = random::<f64>() * 2.0 - 1.0
            }
        }

        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Self {
        Self {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn deserialize(serialized: &str) -> Self {
        let m: Matrix = serde_json::from_str(serialized).unwrap();
        m
    }

    pub fn serialized(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn multiply(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let mut res: Self = Self::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }

        res
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to add matrix of incorrect dimensions");
        }

        let mut res: Self = Self::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        res
    }

    pub fn dot_multiply(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to dot multiply by matrix of incorrect dimensions");
        }

        let mut res: Self = Self::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        res
    }

    pub fn subtract(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to subtract matrix of incorrect dimensions");
        }

        let mut res: Self = Self::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }

    pub fn map(&self, func: &dyn Fn(f64) -> f64) -> Self {
        Self::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| func(value)).collect())
                .collect(),
        )
    }

    pub fn transpose(&self) -> Self {
        let mut res: Self = Self::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }
}
