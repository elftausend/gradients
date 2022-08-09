extern crate proc_macro;
use proc_macro2::{TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, Data, DeriveInput, Field, Fields,
    Ident,
};

#[proc_macro_attribute]
pub fn network(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut modified = proc_macro::TokenStream::new();
    //let mut source = item.into_iter().peekable();
    let input = parse_macro_input!(item as DeriveInput);
    let name = input.ident;
    
    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            _ => panic!("Structs only"),
        },
        _ => panic!("Structs only"),
    };


    proc_macro::TokenStream::from(add_lifetimes(name, fields))
}

fn add_lifetimes(name: Ident, fields: Punctuated<Field, Comma>) -> TokenStream {
    let fields_with_lifetimes = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            let t = &f.ty;
            let type_token = t.into_token_stream();
            let type_token_string = type_token.to_string();
            
            
            if type_token.to_string().starts_with("Linear2") {
                let mut in_out_size = TokenStream::new();
                for token in type_token {
                    if let TokenTree::Literal(lit) = &token {
                        in_out_size.extend(lit.to_token_stream());
                    }

                    if let TokenTree::Punct(pun) = token {
                        if pun.as_char() != ',' {
                            continue;
                        }
                        in_out_size.extend(pun.to_token_stream());
                    }
                    
                }
                
                quote! {#name: Linear2<'a, T, #in_out_size>,}

            } else {
                quote!(#name: #t<'a, T>,)
            }
            
        })
        .collect::<TokenStream>();

    quote! {
        struct #name<'a, T> {
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
        impl<'a, T> GetParam<'a,T> for #name<'a, T> {}
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
        use gradients::{GetParam, Param, Matrix};


        impl<'a, T> Default for #name<'a, T> {
            fn default() -> Self {
                Self { #default_chain }
            }
        }
        impl<'a, T: gradients::number::Float+gradients::CDatatype+gradients::GenericBlas + gradients::CudaTranspose> NeuralNetwork<'a, T> for #name<'a, T> {
            fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
                #forward_chain
            }
            
            fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
                #backward_chain
            }

            fn params(&mut self) -> Vec<Param<'a, T>> {
                #vec
                #params
                #return_vec
            }
        }
    }
}
