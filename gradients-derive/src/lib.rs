extern crate proc_macro;
use proc_macro2::{TokenStream, TokenTree};
use proc_macro_error::{emit_error, proc_macro_error};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, Data, DeriveInput, Field, Fields,
    Ident,
};

#[proc_macro_attribute]
#[proc_macro_error]
pub fn network(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = input.ident;

    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            _ => panic!("The network attribute can be applied on structs only."),
        },
        _ => panic!("The network attribute can be applied on structs only."),
    };

    proc_macro::TokenStream::from(add_lifetimes_derive_net(name, fields))
}

fn add_lifetimes_derive_net(name: Ident, fields: Punctuated<Field, Comma>) -> TokenStream {
    let mut prev_out_size_info = (None, None);

    let fields_with_lifetimes = fields
        .iter()
        .map(|f| {
            let mut in_or_out_size = 0;

            let name = &f.ident;
            let t = &f.ty;
            let type_token = t.into_token_stream();

            if type_token.to_string().starts_with("Linear") {
                let mut in_out_size = TokenStream::new();
                for token in type_token {
                    if let TokenTree::Literal(lit) = &token {
                        let lit_tokens = lit.to_token_stream();

                        in_out_size.extend(lit_tokens.clone());
                        
                        if in_or_out_size == 1 {
                            prev_out_size_info = (Some(lit_tokens.clone()), name.clone());
                        }
                        
                        // comparing the output size with the next input size of the linear layer
                        if let Some(prev_out_size) = &prev_out_size_info.0 {
                            if in_or_out_size == 0 && prev_out_size.to_string() != lit_tokens.to_string() {
                                emit_error! { lit_tokens,
                                    format!("The output and input size of {prev_ident:?} (output size: {prev_out}) and {ident:?} (input size: {input}) do not match.",
                                 
                                        prev_ident=prev_out_size_info.1.as_ref().unwrap().to_string(), 
                                        prev_out=prev_out_size, 
                                        ident=name.as_ref().unwrap().to_string(), 
                                        input=lit_tokens
                                    );                              
                                    note=format!("The input size of {ident:?} must be equal to the output size of {prev_ident:?}.",
                                            ident=name.as_ref().unwrap().to_string(),
                                            prev_ident=prev_out_size_info.1.as_ref().unwrap().to_string(), 
                                    );
                                    help=format!("Set the input size of {ident:?} to {prev_out}.",
                                        ident=name.as_ref().unwrap().to_string(),
                                        prev_out=prev_out_size,
                                    );
                                }
                            }
                        }
                        in_or_out_size += 1;
                    }

                    if let TokenTree::Punct(pun) = token {
                        if pun.as_char() != ',' {
                            continue;
                        }
                        in_out_size.extend(pun.to_token_stream());
                    }
                }

                quote! {#name: Linear<'a, T, #in_out_size, D>,}
            } else {
                quote!(#name: #t<'a, T, D>,)
            }
        })
        .collect::<TokenStream>();

    /*let with_device_chain = fields
    .iter()
    .map(|f| {
        let name = &f.ident;

        quote!(#name: WithDevice::with(device),)
    })
    .collect::<TokenStream>();*/

    quote! {
        use gradients::{NeuralNetwork, Alloc, GraphReturn};
        #[derive(NeuralNetwork)]
        struct #name<'a, T, D: Device> {
            #fields_with_lifetimes
        }
    }
}

#[proc_macro_derive(NoParams)]
pub fn derive_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;
    proc_macro::TokenStream::from(impl_params(name))
}

fn impl_params(name: Ident) -> TokenStream {
    quote! {
        impl<'a, T, D: custos_math::custos::Device> GetParam<'a, T, D> for #name<'a, T, D>
        where
            <D as Device>::Ptr<T, ()>: custos_math::custos::ShallowCopy
        {}
        impl<'a, T, D: custos_math::custos::Device> WithDevice<'a, T, D> for #name<'a, T, D> {}
        impl<'a, T, D: custos_math::custos::Device> #name<'a, T, D> {
            pub fn with_device<'b>(_dev: &'b D) -> #name<'a, T, D> {
                Self::default()
            }
        }
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
    let forward_chain = fields.iter().fold(quote!(&inputs), |acc, f| {
        let name = &f.ident;
        quote!(self.#name.forward(&#acc))
    });

    let default_chain = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            quote!(#name: Default::default(),)
        })
        .collect::<TokenStream>();

    let with_device_chain = fields
        .iter()
        .map(|f| {
            let name = &f.ident;

            quote!(#name: WithDevice::with(device),)
        })
        .collect::<TokenStream>();

    let backward_chain = fields.iter().rev().fold(quote!(&grad), |acc, f| {
        let name = &f.ident;
        quote!(self.#name.backward(&#acc))
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
        use gradients::{GetParam, Param, Matrix, number::{Number, Float}, Device, WithDevice};

        impl<'a, T: Float, D: gradients::Bounds<'a, T>> WithDevice<'a, T, D> for #name<'a, T, D> {
            fn with<'b: 'a>(device: &'b D) -> Self {
                Self { #with_device_chain }
            }
        }

        impl<'a, T: Number, D: Device> Default for #name<'a, T, D> {
            fn default() -> Self {
                Self { #default_chain }
            }
        }

        impl<'a, D: gradients::Bounds<'a, T>, T:Float+gradients::CDatatype+gradients::GenericBlas + gradients::matrix_multiply::MatrixMultiply + gradients::CudaTranspose> NeuralNetwork<'a, T, D> for #name<'a, T, D>
        where <D as Device>::Ptr<T, ()>: gradients::ShallowCopy
        {
            fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D> {
                #forward_chain
            }

            fn backward(&mut self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D> {
                #backward_chain
            }

            fn params(&mut self) -> Vec<Param<'a, T, D>> {
                #vec
                #params
                #return_vec
            }
        }
    }
}
