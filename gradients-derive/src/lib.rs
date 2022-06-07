use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, Data, DeriveInput, Field, Fields,
    Ident,
};

#[proc_macro_derive(NoParams)]
pub fn derive_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    proc_macro::TokenStream::from(impl_params(name))
}

fn impl_params(name: Ident) -> TokenStream {
    quote! {
        impl<T> GetParam<T> for #name<T> {}
    }
}


#[proc_macro_derive(NeuralNetwork)]
pub fn derive_neural_network(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            _ => panic!("Structs only"),
        },
        _ => panic!("Structs only"),
    };

    proc_macro::TokenStream::from(impl_neural_network(name, fields))
}

fn impl_neural_network(name: Ident, fields: Punctuated<Field, Comma>) -> TokenStream {
    let forward_chain = fields.iter().fold(quote!(inputs), |acc, f| {
        let name = &f.ident;
        quote!(self.#name.forward(#acc))
    });

    let default_chain = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            quote!(#name: Default::default(),)
        })
        .collect::<TokenStream>();

    let backward_chain = fields.iter().rev().fold(quote!(grad), |acc, f| {
        let name = &f.ident;
        quote!(self.#name.backward(#acc))
    });

    let vec = quote! {let mut vec = Vec::new();};

    let params = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            quote!(
                if let Some(params) = self.#name.params() {
                    vec.push(params);
                }
             )
        })
        .collect::<TokenStream>();
    let return_vec = quote! {vec};

    quote! {
        use gradients::{GetParam, Param, SampleUniform};
        use custos::Matrix;

        impl<T> Default for #name<T> {
            fn default() -> Self {
                Self { #default_chain }
            }
        }
        impl<T: custos::number::Float+custos::GenericOCL+custos::cpu::TBlas+SampleUniform> NeuralNetwork<T> for #name<T> {
            fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
                #forward_chain
            }
            /*
            fn forward2<L: AsLocDesc<T>>(&mut self, inputs: L) -> Tensor<T> {
                #first_forward
                #forward_chain2
            }
            */
            fn backward(&mut self, grad: Matrix<T>) -> Matrix<T> {
                #backward_chain
            }

            fn params(&mut self) -> Vec<Param<T>> {
                #vec
                #params
                #return_vec
            }
        }
    }
}
