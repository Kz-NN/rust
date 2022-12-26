use std::f64::consts::E;

/**

The struct to create activation and derivated functions.

# Arguments

* `func`: The activation function
* `dfunc`: The derivated of the activation function

*/
#[derive(Clone)]
pub struct Activation<'a> {
    pub func: &'a dyn Fn(f64) -> f64,
    pub dfunc: &'a dyn Fn(f64) -> f64,
}

/**

The sigmoid and his derivated Activation struct.

*/
pub const SIGMOID: Activation = Activation {
    func: &|x| 1.0 / (1.0 + E.powf(-x)),
    dfunc: &|y| y * (1.0 - y),
};

/**

The tanh and his derivated Activation struct.

*/
pub const TANH: Activation = Activation {
    func: &|x| x.tanh(),
    dfunc: &|x| 1.0 - x.powi(2),
};
