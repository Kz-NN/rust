use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a> {
    pub func: &'a dyn Fn(f64) -> f64,
    pub dfunc: &'a dyn Fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    func: &|x| 1.0 / (1.0 + E.powf(-x)),
    dfunc: &|y| y * (1.0 - y),
};

pub const TANH: Activation = Activation {
    func: &|x| x.tanh(),
    dfunc: &|x| 1.0 - x.powi(2),
};
